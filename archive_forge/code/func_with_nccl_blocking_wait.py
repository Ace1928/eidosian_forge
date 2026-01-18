import faulthandler
import logging
import multiprocessing
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial, reduce, wraps
from io import StringIO
from typing import Dict, NamedTuple, Optional, Union
from unittest.mock import patch
import torch
import torch._dynamo.test_case
import torch.cuda.nccl
import torch.distributed as c10d
import torch.nn as nn
from torch.testing._internal.common_utils import (
from torch.testing._internal.distributed.multi_threaded_pg import (
def with_nccl_blocking_wait(func):
    """
    Convenience decorator to set/unset TORCH_NCCL_BLOCKING_WAIT flag. Note that use of
    this decorator will override the setting of TORCH_NCCL_ASYNC_ERROR_HANDLING for
    the particular test. After the test, both TORCH_NCCL_BLOCKING_WAIT and
    TORCH_NCCL_ASYNC_ERROR_HANDLING will be restored to their original values.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            cached_nccl_async_error_handling: Union[str, None] = os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING']
            del os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING']
        except KeyError:
            cached_nccl_async_error_handling = None
        try:
            cached_nccl_blocking_wait: Union[str, None] = os.environ['TORCH_NCCL_BLOCKING_WAIT']
        except KeyError:
            cached_nccl_blocking_wait = None
        finally:
            os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
        try:
            ret = func(*args, **kwargs)
            return ret
        finally:
            if cached_nccl_async_error_handling is not None:
                os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = cached_nccl_async_error_handling
            if cached_nccl_blocking_wait is not None:
                os.environ['TORCH_NCCL_BLOCKING_WAIT'] = cached_nccl_blocking_wait
    return wrapper