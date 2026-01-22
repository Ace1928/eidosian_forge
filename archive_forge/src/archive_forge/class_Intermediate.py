import asyncio
import logging
from typing import Tuple
import aiohttp
import click
from starlette.responses import StreamingResponse
from ray import serve
from ray.serve._private.benchmarks.common import run_throughput_benchmark
from ray.serve.handle import DeploymentHandle, RayServeHandle
@serve.deployment(ray_actor_options={'num_cpus': 0})
class Intermediate:

    def __init__(self, downstream: RayServeHandle):
        logging.getLogger('ray.serve').setLevel(logging.WARNING)
        self._h: DeploymentHandle = downstream.options(stream=True, use_new_handle_api=True)

    async def stream(self):
        async for token in self._h.stream.remote():
            yield token

    def __call__(self, *args):
        return StreamingResponse(self.stream())