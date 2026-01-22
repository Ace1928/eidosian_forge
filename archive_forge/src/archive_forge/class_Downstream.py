import asyncio
import logging
from pprint import pprint
from typing import Dict, Union
import aiohttp
from starlette.requests import Request
import ray
from ray import serve
from ray.serve._private.benchmarks.common import run_throughput_benchmark
from ray.serve.handle import RayServeHandle
@serve.deployment(num_replicas=num_replicas, max_concurrent_queries=max_concurrent_queries)
class Downstream:

    def __init__(self):
        logging.getLogger('ray.serve').setLevel(logging.WARNING)

    @serve.batch(max_batch_size=max_batch_size)
    async def batch(self, reqs):
        return [b'ok'] * len(reqs)

    async def __call__(self, req: Union[bytes, Request]):
        if max_batch_size > 1:
            return await self.batch(req)
        else:
            return b'ok'