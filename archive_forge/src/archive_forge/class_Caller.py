import asyncio
import logging
from typing import Tuple
import click
from ray import serve
from ray.serve._private.benchmarks.common import run_throughput_benchmark
from ray.serve.handle import DeploymentHandle, RayServeHandle
@serve.deployment
class Caller:

    def __init__(self, downstream: RayServeHandle, *, tokens_per_request: int, batch_size: int, num_trials: int, trial_runtime: float):
        logging.getLogger('ray.serve').setLevel(logging.WARNING)
        self._h: DeploymentHandle = downstream.options(use_new_handle_api=True, stream=True)
        self._tokens_per_request = tokens_per_request
        self._batch_size = batch_size
        self._num_trials = num_trials
        self._trial_runtime = trial_runtime

    async def _consume_single_stream(self):
        async for _ in self._h.stream.remote():
            pass

    async def _do_single_batch(self):
        await asyncio.gather(*[self._consume_single_stream() for _ in range(self._batch_size)])

    async def run_benchmark(self) -> Tuple[float, float]:
        return await run_throughput_benchmark(fn=self._do_single_batch, multiplier=self._batch_size * self._tokens_per_request, num_trials=self._num_trials, trial_runtime=self._trial_runtime)