from contextlib import contextmanager
from typing import Any, List, Tuple, cast
import random
import torch
import time
from torch.utils.benchmark import Timer
def time_cpu(fn, inputs, test_runs):
    s = time.perf_counter()
    for _ in range(test_runs):
        fn(*inputs)
    e = time.perf_counter()
    return (e - s) / test_runs * 1000