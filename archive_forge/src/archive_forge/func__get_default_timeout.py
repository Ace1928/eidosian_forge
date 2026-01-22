import itertools
import collections.abc
import contextlib
import hashlib
import io
import logging
import os
import pickle
import sys
import time
import warnings
from collections import namedtuple
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import torch
from torch._C._distributed_c10d import (
from .constants import default_pg_timeout, default_pg_nccl_timeout
from .c10d_logger import _exception_logger, _time_logger
from .rendezvous import register_rendezvous_handler, rendezvous  # noqa: F401
def _get_default_timeout(backend: Backend) -> timedelta:
    if backend == Backend.NCCL:
        if not isinstance(default_pg_nccl_timeout, timedelta):
            warnings.warn('Attempted to get default timeout for nccl backend, but NCCL support is not compiled')
            return default_pg_timeout
        return default_pg_nccl_timeout
    else:
        return default_pg_timeout