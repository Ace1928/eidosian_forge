import asyncio
import logging
from ray._private.ray_microbenchmark_helpers import timeit
from ray._private.ray_client_microbenchmark import main as client_microbenchmark_main
import numpy as np
import multiprocessing
import ray
@ray.remote
class AsyncActor:

    async def small_value(self):
        return b'ok'

    async def small_value_with_arg(self, x):
        return b'ok'

    async def small_value_batch(self, n):
        await asyncio.wait([small_value.remote() for _ in range(n)])