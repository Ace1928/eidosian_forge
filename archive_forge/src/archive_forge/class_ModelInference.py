import asyncio
import json
import logging
import time
from random import random
from typing import Callable, Dict
import aiohttp
import numpy as np
import pandas as pd
from grpc import aio
from starlette.requests import Request
import ray
from ray import serve
from ray.serve._private.common import RequestProtocol
from ray.serve.config import gRPCOptions
from ray.serve.generated import serve_pb2, serve_pb2_grpc
from ray.serve.handle import RayServeHandle
@serve.deployment(num_replicas=num_replicas, max_concurrent_queries=max_concurrent_queries)
class ModelInference:

    def __init__(self):
        logging.getLogger('ray.serve').setLevel(logging.WARNING)
        self._model = np.random.randn(data_size, data_size)

    async def __call__(self, processed: np.ndarray) -> float:
        model_output = np.dot(processed, self._model)
        return sum(model_output)