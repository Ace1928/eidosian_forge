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
class LogstreamServicerProxy(ray_client_pb2_grpc.RayletLogStreamerServicer):

    def __init__(self, proxy_manager: ProxyManager):
        super().__init__()
        self.proxy_manager = proxy_manager

    def Logstream(self, request_iterator, context):
        request_iterator = RequestIteratorProxy(request_iterator)
        client_id = _get_client_id_from_context(context)
        if client_id == '':
            return
        logger.debug(f'New logstream connection from client {client_id}: ')
        channel = None
        for i in range(LOGSTREAM_RETRIES):
            channel = self.proxy_manager.get_channel(client_id)
            if channel is not None:
                break
            logger.warning(f'Retrying Logstream connection. {i + 1} attempts failed.')
            time.sleep(LOGSTREAM_RETRY_INTERVAL_SEC)
        if channel is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f'Logstream proxy failed to connect. Channel for client {client_id} not found.')
            return None
        stub = ray_client_pb2_grpc.RayletLogStreamerStub(channel)
        resp_stream = stub.Logstream(request_iterator, metadata=[('client_id', client_id)])
        try:
            for resp in resp_stream:
                yield resp
        except Exception:
            logger.exception('Proxying Logstream failed!')