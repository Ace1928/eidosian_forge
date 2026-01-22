import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def _check_inside_actor():
    """Check if currently it is inside a Ray actor/task."""
    worker = ray._private.worker.global_worker
    if worker.mode == ray.WORKER_MODE:
        return
    else:
        raise RuntimeError('The collective APIs shall be only used inside a Ray actor or task.')