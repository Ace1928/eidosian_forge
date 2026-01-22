import logging
from typing import Optional, Tuple
import concurrent.futures
import ray.dashboard.modules.log.log_utils as log_utils
import ray.dashboard.modules.log.log_consts as log_consts
import ray.dashboard.utils as dashboard_utils
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray._private.ray_constants import env_integer
import asyncio
import grpc
import io
import os
from pathlib import Path
from ray.core.generated import reporter_pb2
from ray.core.generated import reporter_pb2_grpc
from ray._private.ray_constants import (
class LogAgent(dashboard_utils.DashboardAgentModule):

    def __init__(self, dashboard_agent):
        super().__init__(dashboard_agent)
        log_utils.register_mimetypes()
        routes.static('/logs', self._dashboard_agent.log_dir, show_index=True)

    async def run(self, server):
        pass

    @staticmethod
    def is_minimal_module():
        return False