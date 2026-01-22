import atexit
import json
import logging
import socket
import sys
import time
import traceback
from concurrent import futures
from dataclasses import dataclass
from itertools import chain
import urllib
from threading import Event, Lock, RLock, Thread
from typing import Callable, Dict, List, Optional, Tuple
import grpc
import psutil
import ray
import ray.core.generated.agent_manager_pb2 as agent_manager_pb2
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
import ray.core.generated.runtime_env_agent_pb2 as runtime_env_agent_pb2
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.parameter import RayParams
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.services import ProcessInfo, start_ray_client_server
from ray._private.tls_utils import add_port_to_grpc_server
from ray._private.utils import detect_fate_sharing_support
from ray.cloudpickle.compat import pickle
from ray.job_config import JobConfig
from ray.util.client.common import (
from ray.util.client.server.dataservicer import _get_reconnecting_from_context
def _create_runtime_env(self, serialized_runtime_env: str, runtime_env_config: str, specific_server: SpecificServer):
    """Increase the runtime_env reference by sending an RPC to the agent.

        Includes retry logic to handle the case when the agent is
        temporarily unreachable (e.g., hasn't been started up yet).
        """
    logger.info(f'Increasing runtime env reference for ray_client_server_{specific_server.port}.Serialized runtime env is {serialized_runtime_env}.')
    assert len(self._runtime_env_agent_address) > 0, 'runtime_env_agent_address not set'
    create_env_request = runtime_env_agent_pb2.GetOrCreateRuntimeEnvRequest(serialized_runtime_env=serialized_runtime_env, runtime_env_config=runtime_env_config, job_id=f'ray_client_server_{specific_server.port}'.encode('utf-8'), source_process='client_server')
    retries = 0
    max_retries = 5
    wait_time_s = 0.5
    last_exception = None
    while retries <= max_retries:
        try:
            url = urllib.parse.urljoin(self._runtime_env_agent_address, '/get_or_create_runtime_env')
            data = create_env_request.SerializeToString()
            req = urllib.request.Request(url, data=data, method='POST')
            req.add_header('Content-Type', 'application/octet-stream')
            response = urllib.request.urlopen(req, timeout=None)
            response_data = response.read()
            r = runtime_env_agent_pb2.GetOrCreateRuntimeEnvReply()
            r.ParseFromString(response_data)
            if r.status == agent_manager_pb2.AgentRpcStatus.AGENT_RPC_STATUS_OK:
                return r.serialized_runtime_env_context
            elif r.status == agent_manager_pb2.AgentRpcStatus.AGENT_RPC_STATUS_FAILED:
                raise RuntimeError(f'Failed to create runtime_env for Ray client server, it is caused by:\n{r.error_message}')
            else:
                assert False, f'Unknown status: {r.status}.'
        except urllib.error.URLError as e:
            last_exception = e
            logger.warning(f'GetOrCreateRuntimeEnv request failed: {e}. Retrying after {wait_time_s}s. {max_retries - retries} retries remaining.')
        time.sleep(wait_time_s)
        retries += 1
        wait_time_s *= 2
    raise TimeoutError(f'GetOrCreateRuntimeEnv request failed after {max_retries} attempts. Last exception: {last_exception}')