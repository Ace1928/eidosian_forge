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
def _on_callback_completed(self, trigger_at: int):
    """Called after a single callback is completed.

        When the total number of callback completed equals to trigger_at,
        _poll_next() will be called. This is designed to make sure we only
        _poll_next() after all the state callbacks completed. This is a
        way to serialize the callback invocations between object versions.
        """
    self._callbacks_processed_count += 1
    if self._callbacks_processed_count == trigger_at:
        self._poll_next()