import asyncio
import logging
import os
import random
from asyncio.events import AbstractEventLoop
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, DefaultDict, Dict, Optional, Set, Tuple, Union
import ray
from ray._private.utils import get_or_create_event_loop
from ray.serve._private.common import ReplicaName
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import format_actor_name
from ray.serve.generated.serve_pb2 import ActorNameList
from ray.serve.generated.serve_pb2 import EndpointInfo as EndpointInfoProto
from ray.serve.generated.serve_pb2 import EndpointSet, LongPollRequest, LongPollResult
from ray.serve.generated.serve_pb2 import UpdatedObject as UpdatedObjectProto
from ray.util import metrics
def _process_update(self, updates: Dict[str, UpdatedObject]):
    if isinstance(updates, ray.exceptions.RayActorError):
        logger.debug('LongPollClient failed to connect to host. Shutting down.')
        self.is_running = False
        return
    if isinstance(updates, ConnectionError):
        logger.warning('LongPollClient connection failed, shutting down.')
        self.is_running = False
        return
    if isinstance(updates, ray.exceptions.RayTaskError):
        logger.error('LongPollHost errored\n' + updates.traceback_str)
        self._schedule_to_event_loop(self._poll_next)
        return
    if updates == LongPollState.TIME_OUT:
        logger.debug('LongPollClient polling timed out. Retrying.')
        self._schedule_to_event_loop(self._poll_next)
        return
    logger.debug(f'LongPollClient {self} received updates for keys: {list(updates.keys())}.', extra={'log_to_stderr': False})
    for key, update in updates.items():
        self.snapshot_ids[key] = update.snapshot_id
        callback = self.key_listeners[key]

        def chained(callback=callback, arg=update.object_snapshot):
            callback(arg)
            self._on_callback_completed(trigger_at=len(updates))
        self._schedule_to_event_loop(chained)