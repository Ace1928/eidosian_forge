import logging
import os
import socket
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import exception_cause, skip_exceptions
from ray.types import ObjectRef
from ray.util.placement_group import PlacementGroup
def add_workers(self, num_workers: int):
    """Adds ``num_workers`` to this WorkerGroup.

        Note: Adding workers when the cluster/placement group is at capacity
        may lead to undefined hanging behavior. If you are attempting to
        replace existing workers in the WorkerGroup, remove_workers() should
        be called first.

        Args:
            num_workers: The number of workers to add.
        """
    new_actors = []
    new_actor_metadata = []
    for _ in range(num_workers):
        actor = self._remote_cls.options(placement_group=self._placement_group).remote(*self._actor_cls_args, **self._actor_cls_kwargs)
        new_actors.append(actor)
        new_actor_metadata.append(actor._RayTrainWorker__execute.options(name='_RayTrainWorker__execute.construct_metadata').remote(construct_metadata))
    metadata = ray.get(new_actor_metadata)
    for i in range(len(new_actors)):
        self.workers.append(Worker(actor=new_actors[i], metadata=metadata[i]))