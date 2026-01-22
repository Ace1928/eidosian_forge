import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import pathlib
import sys
import signal
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services
import ray._private.utils
import ray.dashboard.consts as dashboard_consts
import ray.dashboard.utils as dashboard_utils
from ray._raylet import GcsClient
from ray._private.process_watcher import create_check_raylet_task
from ray._private.gcs_utils import GcsAioClient
from ray._private.ray_logging import (
from ray.experimental.internal_kv import (
from ray._private.ray_constants import AGENT_GRPC_MAX_MESSAGE_LENGTH
class DashboardAgent:

    def __init__(self, node_ip_address, dashboard_agent_port, gcs_address, minimal, metrics_export_port=None, node_manager_port=None, listen_port=ray_constants.DEFAULT_DASHBOARD_AGENT_LISTEN_PORT, disable_metrics_collection: bool=False, *, object_store_name: str, raylet_name: str, log_dir: str, temp_dir: str, session_dir: str, logging_params: dict, agent_id: int, session_name: str):
        """Initialize the DashboardAgent object."""
        self.ip = node_ip_address
        self.minimal = minimal
        assert gcs_address is not None
        self.gcs_address = gcs_address
        self.temp_dir = temp_dir
        self.session_dir = session_dir
        self.log_dir = log_dir
        self.dashboard_agent_port = dashboard_agent_port
        self.metrics_export_port = metrics_export_port
        self.node_manager_port = node_manager_port
        self.listen_port = listen_port
        self.object_store_name = object_store_name
        self.raylet_name = raylet_name
        self.logging_params = logging_params
        self.node_id = os.environ['RAY_NODE_ID']
        self.metrics_collection_disabled = disable_metrics_collection
        self.agent_id = agent_id
        self.session_name = session_name
        self.server = None
        self.http_server = None
        self.gcs_client = GcsClient(address=self.gcs_address)
        _initialize_internal_kv(self.gcs_client)
        assert _internal_kv_initialized()
        self.gcs_aio_client = GcsAioClient(address=self.gcs_address)
        if not self.minimal:
            self._init_non_minimal()

    def _init_non_minimal(self):
        from ray._private.gcs_pubsub import GcsAioPublisher
        self.aio_publisher = GcsAioPublisher(address=self.gcs_address)
        try:
            from grpc import aio as aiogrpc
        except ImportError:
            from grpc.experimental import aio as aiogrpc
        if sys.version_info.major >= 3 and sys.version_info.minor >= 10:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=DeprecationWarning)
                aiogrpc.init_grpc_aio()
        else:
            aiogrpc.init_grpc_aio()
        self.server = aiogrpc.server(options=(('grpc.so_reuseport', 0), ('grpc.max_send_message_length', AGENT_GRPC_MAX_MESSAGE_LENGTH), ('grpc.max_receive_message_length', AGENT_GRPC_MAX_MESSAGE_LENGTH)))
        grpc_ip = '127.0.0.1' if self.ip == '127.0.0.1' else '0.0.0.0'
        try:
            self.grpc_port = ray._private.tls_utils.add_port_to_grpc_server(self.server, f'{grpc_ip}:{self.dashboard_agent_port}')
        except Exception:
            logger.exception('Failed to add port to grpc server. Agent will stay alive but disable the grpc service.')
            self.server = None
            self.grpc_port = None
        else:
            logger.info('Dashboard agent grpc address: %s:%s', grpc_ip, self.grpc_port)

    async def _configure_http_server(self, modules):
        from ray.dashboard.http_server_agent import HttpServerAgent
        http_server = HttpServerAgent(self.ip, self.listen_port)
        await http_server.start(modules)
        return http_server

    def _load_modules(self):
        """Load dashboard agent modules."""
        modules = []
        agent_cls_list = dashboard_utils.get_all_modules(dashboard_utils.DashboardAgentModule)
        for cls in agent_cls_list:
            logger.info('Loading %s: %s', dashboard_utils.DashboardAgentModule.__name__, cls)
            c = cls(self)
            modules.append(c)
        logger.info('Loaded %d modules.', len(modules))
        return modules

    @property
    def http_session(self):
        assert self.http_server, 'Accessing unsupported API (HttpServerAgent) in a minimal ray.'
        return self.http_server.http_session

    @property
    def publisher(self):
        assert self.aio_publisher, 'Accessing unsupported API (GcsAioPublisher) in a minimal ray.'
        return self.aio_publisher

    def get_node_id(self) -> str:
        return self.node_id

    async def run(self):
        if self.server:
            await self.server.start()
        modules = self._load_modules()
        if not self.minimal:
            try:
                self.http_server = await self._configure_http_server(modules)
            except Exception:
                logger.exception('Failed to start http server. Agent will stay alive but disable the http service.')
        http_port = -1 if not self.http_server else self.http_server.http_port
        grpc_port = -1 if not self.server else self.grpc_port
        await self.gcs_aio_client.internal_kv_put(f'{dashboard_consts.DASHBOARD_AGENT_PORT_PREFIX}{self.node_id}'.encode(), json.dumps([http_port, grpc_port]).encode(), True, namespace=ray_constants.KV_NAMESPACE_DASHBOARD)
        tasks = [m.run(self.server) for m in modules]
        if sys.platform not in ['win32', 'cygwin']:

            def callback(msg):
                logger.info(f'Terminated Raylet: ip={self.ip}, node_id={self.node_id}. {msg}')
            check_parent_task = create_check_raylet_task(self.log_dir, self.gcs_address, callback, loop)
            tasks.append(check_parent_task)
        await asyncio.gather(*tasks)
        if self.server:
            await self.server.wait_for_termination()
        else:
            while True:
                await asyncio.sleep(3600)
        if self.http_server:
            await self.http_server.cleanup()