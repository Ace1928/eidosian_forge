import base64
import functools
import gc
import inspect
import json
import logging
import math
import pickle
import queue
import threading
import time
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Dict, List, Optional, Set, Union
import grpc
import ray
import ray._private.state
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray import cloudpickle
from ray._private import ray_constants
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.ray_constants import env_integer
from ray._private.ray_logging import setup_logger
from ray._private.services import canonicalize_bootstrap_address_or_die
from ray._private.tls_utils import add_port_to_grpc_server
from ray.job_config import JobConfig
from ray.util.client.common import (
from ray.util.client.server.dataservicer import DataServicer
from ray.util.client.server.logservicer import LogstreamServicer
from ray.util.client.server.proxier import serve_proxier
from ray.util.client.server.server_pickler import dumps_from_server, loads_from_client
from ray.util.client.server.server_stubs import current_server
def _release_actors(self, client_id):
    if client_id not in self.actor_owners:
        logger.debug(f'Releasing client with no actors: {client_id}')
        return
    count = 0
    actors_to_remove = self.actor_owners.pop(client_id)
    for id_bytes in actors_to_remove:
        count += 1
        if self._can_remove_actor_ref(id_bytes):
            logger.debug(f'Deleting reference to actor {id_bytes.hex()}')
            del self.actor_refs[id_bytes]
    logger.debug(f'Released all {count} actors for client: {client_id}')