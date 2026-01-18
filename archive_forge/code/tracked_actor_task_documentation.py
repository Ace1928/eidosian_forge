from typing import Any, Callable, Optional
from ray.air.execution._internal.tracked_actor import TrackedActor
Actor task tracked by a Ray event manager.

    This container class is used to define callbacks to be invoked when
    the task resolves, errors, or times out.

    Note:
        Objects of this class are returned by the :class:`RayActorManager`.
        This class should not be instantiated manually.

    Args:
        tracked_actor: Tracked actor object this task is scheduled on.
        on_result: Callback to invoke when the task resolves.
        on_error: Callback to invoke when the task fails.

    Example:

        .. code-block:: python

            tracked_futures = actor_manager.schedule_actor_tasks(
                actor_manager.live_actors,
                "foo",
                on_result=lambda actor, result: print(result)
                )

    