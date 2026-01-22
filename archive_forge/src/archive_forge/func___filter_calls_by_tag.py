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
def __filter_calls_by_tag(self, tags) -> Tuple[List[ray.ObjectRef], List[ActorHandle], List[str]]:
    """Return all the in flight requests that match the given tags.

        Args:
            tags: A str or a list of str. If tags is empty, return all the in flight

        Returns:
            A tuple of corresponding (remote_calls, remote_actor_ids, valid_tags)

        """
    if isinstance(tags, str):
        tags = {tags}
    elif isinstance(tags, (list, tuple)):
        tags = set(tags)
    else:
        raise ValueError(f'tags must be either a str or a list of str, got {type(tags)}.')
    remote_calls = []
    remote_actor_ids = []
    valid_tags = []
    for call, (tag, actor_id) in self.__in_flight_req_to_actor_id.items():
        if not len(tags) or tag in tags:
            remote_calls.append(call)
            remote_actor_ids.append(actor_id)
            valid_tags.append(tag)
    return (remote_calls, remote_actor_ids, valid_tags)