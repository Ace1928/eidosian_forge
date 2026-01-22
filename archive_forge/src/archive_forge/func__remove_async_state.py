from collections import defaultdict
import copy
from dataclasses import dataclass
import logging
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError, RayError, RayTaskError
from ray.rllib.utils.typing import T
from ray.util.annotations import DeveloperAPI
def _remove_async_state(self, actor_id: int):
    """Remove internal async state of for a given actor.

        This is called when an actor is removed from the pool or being marked
        unhealthy.

        Args:
            actor_id: The id of the actor.
        """
    reqs_to_be_removed = [req for req, id in self.__in_flight_req_to_actor_id.items() if id == actor_id]
    for req in reqs_to_be_removed:
        del self.__in_flight_req_to_actor_id[req]