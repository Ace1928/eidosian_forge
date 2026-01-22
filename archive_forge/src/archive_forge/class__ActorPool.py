import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_operator import MapOperator, _map_task
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.execution.util import locality_string
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
class _ActorPool:
    """A pool of actors for map task execution.

    This class is in charge of tracking the number of in-flight tasks per actor,
    providing the least heavily loaded actor to the operator, and killing idle
    actors when the operator is done submitting work to the pool.
    """

    def __init__(self, max_tasks_in_flight: int=DEFAULT_MAX_TASKS_IN_FLIGHT):
        self._max_tasks_in_flight = max_tasks_in_flight
        self._num_tasks_in_flight: Dict[ray.actor.ActorHandle, int] = {}
        self._actor_locations: Dict[ray.actor.ActorHandle, str] = {}
        self._pending_actors: Dict[ObjectRef, ray.actor.ActorHandle] = {}
        self._should_kill_idle_actors = False
        self._locality_hits: int = 0
        self._locality_misses: int = 0

    def add_pending_actor(self, actor: ray.actor.ActorHandle, ready_ref: ray.ObjectRef):
        """Adds a pending actor to the pool.

        This actor won't be pickable until it is marked as running via a
        pending_to_running() call.

        Args:
            actor: The not-yet-ready actor to add as pending to the pool.
            ready_ref: The ready future for the actor.
        """
        assert not self._should_kill_idle_actors
        self._pending_actors[ready_ref] = actor

    def pending_to_running(self, ready_ref: ray.ObjectRef) -> bool:
        """Mark the actor corresponding to the provided ready future as running, making
        the actor pickable.

        Args:
            ready_ref: The ready future for the actor that we wish to mark as running.

        Returns:
            Whether the actor was still pending. This can return False if the actor had
            already been killed.
        """
        if ready_ref not in self._pending_actors:
            return False
        actor = self._pending_actors.pop(ready_ref)
        self._num_tasks_in_flight[actor] = 0
        self._actor_locations[actor] = ray.get(ready_ref)
        return True

    def pick_actor(self, locality_hint: Optional[RefBundle]=None) -> Optional[ray.actor.ActorHandle]:
        """Picks an actor for task submission based on busyness and locality.

        None will be returned if all actors are either at capacity (according to
        max_tasks_in_flight) or are still pending.

        Args:
            locality_hint: Try to pick an actor that is local for this bundle.
        """
        if not self._num_tasks_in_flight:
            return None
        if locality_hint:
            preferred_loc = self._get_location(locality_hint)
        else:
            preferred_loc = None

        def penalty_key(actor):
            """Returns the key that should be minimized for the best actor.

            We prioritize valid actors, those with argument locality, and those that
            are not busy, in that order.
            """
            busyness = self._num_tasks_in_flight[actor]
            invalid = busyness >= self._max_tasks_in_flight
            requires_remote_fetch = self._actor_locations[actor] != preferred_loc
            return (invalid, requires_remote_fetch, busyness)
        actor = min(self._num_tasks_in_flight.keys(), key=penalty_key)
        if self._num_tasks_in_flight[actor] >= self._max_tasks_in_flight:
            return None
        if locality_hint:
            if self._actor_locations[actor] == preferred_loc:
                self._locality_hits += 1
            else:
                self._locality_misses += 1
        self._num_tasks_in_flight[actor] += 1
        return actor

    def return_actor(self, actor: ray.actor.ActorHandle):
        """Returns the provided actor to the pool."""
        assert actor in self._num_tasks_in_flight
        assert self._num_tasks_in_flight[actor] > 0
        self._num_tasks_in_flight[actor] -= 1
        if self._should_kill_idle_actors and self._num_tasks_in_flight[actor] == 0:
            self._kill_running_actor(actor)

    def get_pending_actor_refs(self) -> List[ray.ObjectRef]:
        return list(self._pending_actors.keys())

    def num_total_actors(self) -> int:
        """Return the total number of actors managed by this pool, including pending
        actors
        """
        return self.num_pending_actors() + self.num_running_actors()

    def num_running_actors(self) -> int:
        """Return the number of running actors in the pool."""
        return len(self._num_tasks_in_flight)

    def num_idle_actors(self) -> int:
        """Return the number of idle actors in the pool."""
        return sum((1 if tasks_in_flight == 0 else 0 for tasks_in_flight in self._num_tasks_in_flight.values()))

    def num_pending_actors(self) -> int:
        """Return the number of pending actors in the pool."""
        return len(self._pending_actors)

    def num_free_slots(self) -> int:
        """Return the number of free slots for task execution."""
        if not self._num_tasks_in_flight:
            return 0
        return sum((max(0, self._max_tasks_in_flight - num_tasks_in_flight) for num_tasks_in_flight in self._num_tasks_in_flight.values()))

    def num_active_actors(self) -> int:
        """Return the number of actors in the pool with at least one active task."""
        return sum((1 if num_tasks_in_flight > 0 else 0 for num_tasks_in_flight in self._num_tasks_in_flight.values()))

    def kill_inactive_actor(self) -> bool:
        """Kills a single pending or idle actor, if any actors are pending/idle.

        Returns whether an inactive actor was actually killed.
        """
        killed = self._maybe_kill_pending_actor()
        if not killed:
            killed = self._maybe_kill_idle_actor()
        return killed

    def _maybe_kill_pending_actor(self) -> bool:
        if self._pending_actors:
            self._kill_pending_actor(next(iter(self._pending_actors.keys())))
            return True
        return False

    def _maybe_kill_idle_actor(self) -> bool:
        for actor, tasks_in_flight in self._num_tasks_in_flight.items():
            if tasks_in_flight == 0:
                self._kill_running_actor(actor)
                return True
        return False

    def kill_all_inactive_actors(self):
        """Kills all currently inactive actors and ensures that all actors that become
        idle in the future will be eagerly killed.

        This is called once the operator is done submitting work to the pool, and this
        function is idempotent. Adding new pending actors after calling this function
        will raise an error.
        """
        self._kill_all_pending_actors()
        self._kill_all_idle_actors()

    def kill_all_actors(self):
        """Kills all actors, including running/active actors.

        This is called once the operator is shutting down.
        """
        self._kill_all_pending_actors()
        self._kill_all_running_actors()

    def _kill_all_pending_actors(self):
        pending_actor_refs = list(self._pending_actors.keys())
        for ref in pending_actor_refs:
            self._kill_pending_actor(ref)

    def _kill_all_idle_actors(self):
        idle_actors = [actor for actor, tasks_in_flight in self._num_tasks_in_flight.items() if tasks_in_flight == 0]
        for actor in idle_actors:
            self._kill_running_actor(actor)
        self._should_kill_idle_actors = True

    def _kill_all_running_actors(self):
        actors = list(self._num_tasks_in_flight.keys())
        for actor in actors:
            self._kill_running_actor(actor)

    def _kill_running_actor(self, actor: ray.actor.ActorHandle):
        """Kill the provided actor and remove it from the pool."""
        ray.kill(actor)
        del self._num_tasks_in_flight[actor]

    def _kill_pending_actor(self, ready_ref: ray.ObjectRef):
        """Kill the provided pending actor and remove it from the pool."""
        actor = self._pending_actors.pop(ready_ref)
        ray.kill(actor)

    def _get_location(self, bundle: RefBundle) -> Optional[NodeIdStr]:
        """Ask Ray for the node id of the given bundle.

        This method may be overriden for testing.

        Returns:
            A node id associated with the bundle, or None if unknown.
        """
        return bundle.get_cached_location()