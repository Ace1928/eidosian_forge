import asyncio
import logging
from ray._private.ray_microbenchmark_helpers import timeit
from ray._private.ray_client_microbenchmark import main as client_microbenchmark_main
import numpy as np
import multiprocessing
import ray
def wait_multiple_refs():
    num_objs = 1000
    not_ready = [small_value.remote() for _ in range(num_objs)]
    fetch_local = True
    for _ in range(num_objs):
        _ready, not_ready = ray.wait(not_ready, fetch_local=fetch_local)
        if fetch_local:
            fetch_local = False