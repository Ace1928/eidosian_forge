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
@dataclass
class SpecificServer:
    port: int
    process_handle_future: futures.Future
    channel: 'grpc._channel.Channel'

    def is_ready(self) -> bool:
        """Check if the server is ready or not (doesn't block)."""
        return self.process_handle_future.done()

    def wait_ready(self, timeout: Optional[float]=None) -> None:
        """
        Wait for the server to actually start up.
        """
        res = self.process_handle_future.result(timeout=timeout)
        if res is None:
            raise RuntimeError('Server startup failed.')

    def poll(self) -> Optional[int]:
        """Check if the process has exited."""
        try:
            proc = self.process_handle_future.result(timeout=0.1)
            if proc is not None:
                return proc.process.poll()
        except futures.TimeoutError:
            return

    def kill(self) -> None:
        """Try to send a KILL signal to the process."""
        try:
            proc = self.process_handle_future.result(timeout=0.1)
            if proc is not None:
                proc.process.kill()
        except futures.TimeoutError:
            pass

    def set_result(self, proc: Optional[ProcessInfo]) -> None:
        """Set the result of the internal future if it is currently unset."""
        if not self.is_ready():
            self.process_handle_future.set_result(proc)