import base64
import json
import logging
import os
import tempfile
import threading
import time
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import grpc
import ray._private.tls_utils
import ray.cloudpickle as cloudpickle
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private.ray_constants import DEFAULT_CLIENT_RECONNECT_GRACE_PERIOD
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray.cloudpickle.compat import pickle
from ray.exceptions import GetTimeoutError
from ray.job_config import JobConfig
from ray.util.client.client_pickler import dumps_from_client, loads_from_server
from ray.util.client.common import (
from ray.util.client.dataclient import DataClient
from ray.util.client.logsclient import LogstreamClient
from ray.util.debug import log_once
def _get_object_iterator(self, req: ray_client_pb2.GetRequest, *args, **kwargs) -> Any:
    """
        Calls the stub for GetObject on the underlying server stub. If a
        recoverable error occurs while streaming the response, attempts
        to retry the get starting from the first chunk that hasn't been
        received.
        """
    last_seen_chunk = -1
    while not self._in_shutdown:
        req.start_chunk_id = last_seen_chunk + 1
        try:
            for chunk in self.server.GetObject(req, *args, **kwargs):
                if chunk.chunk_id <= last_seen_chunk:
                    logger.debug(f'Received a repeated chunk {chunk.chunk_id} from request {req.req_id}.')
                    continue
                if last_seen_chunk + 1 != chunk.chunk_id:
                    raise RuntimeError(f'Received chunk {chunk.chunk_id} when we expected {self.last_seen_chunk + 1}')
                last_seen_chunk = chunk.chunk_id
                yield chunk
                if last_seen_chunk == chunk.total_chunks - 1:
                    return
            return
        except grpc.RpcError as e:
            if self._can_reconnect(e):
                time.sleep(0.5)
                continue
            raise
        except ValueError:
            time.sleep(0.5)
            continue
    raise ConnectionError('Client is shutting down.')