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
gRPC entrypoint.

            It parses the request, normalize the data, and send to model for inference.
            