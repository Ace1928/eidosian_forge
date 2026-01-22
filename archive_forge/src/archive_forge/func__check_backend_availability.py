import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def _check_backend_availability(backend: types.Backend):
    """Check whether the backend is available."""
    if backend == types.Backend.GLOO:
        if not gloo_available():
            raise RuntimeError('GLOO is not available.')
    elif backend == types.Backend.NCCL:
        if not nccl_available():
            raise RuntimeError('NCCL is not available.')