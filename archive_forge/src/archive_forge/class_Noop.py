import asyncio
import logging
import click
import pandas as pd
import requests
from ray import serve
from ray.serve._private.benchmarks.common import run_latency_benchmark
@serve.deployment
class Noop:

    def __init__(self):
        logging.getLogger('ray.serve').setLevel(logging.WARNING)

    def __call__(self, _):
        return b''