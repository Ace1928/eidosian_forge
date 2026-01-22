from collections import defaultdict
from ray.util.client.server.server_pickler import loads_from_client
import ray
import logging
import grpc
from queue import Queue
import sys
from typing import Any, Dict, Iterator, TYPE_CHECKING, Union
from threading import Event, Lock, Thread
import time
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.client.common import (
from ray.util.client import CURRENT_PROTOCOL_VERSION
from ray.util.debug import log_once
from ray._private.client_mode_hook import disable_client_hook
def _build_connection_response(self):
    with self.clients_lock:
        cur_num_clients = self.num_clients
    return ray_client_pb2.ConnectionInfoResponse(num_clients=cur_num_clients, python_version='{}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]), ray_version=ray.__version__, ray_commit=ray.__commit__, protocol_version=CURRENT_PROTOCOL_VERSION)