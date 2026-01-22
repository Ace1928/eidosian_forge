import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import inspect
import io
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
from unittest.mock import MagicMock
import expecttest
import numpy as np
import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch.nn import (
from torch.onnx import (
from torch.testing import make_tensor
from torch.testing._comparison import (
from torch.testing._comparison import not_close_error_metas
from torch.testing._internal.common_dtype import get_all_dtypes
import torch.utils._pytree as pytree
from .composite_compliance import no_dispatch
class CudaMemoryLeakCheck:

    def __init__(self, testcase, name=None):
        self.name = testcase.id() if name is None else name
        self.testcase = testcase
        from torch.testing._internal.common_cuda import initialize_cuda_context_rng
        initialize_cuda_context_rng()

    def __enter__(self):
        self.caching_allocator_befores = []
        self.driver_befores = []
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            caching_allocator_mem_allocated = torch.cuda.memory_allocated(i)
            if caching_allocator_mem_allocated > 0:
                gc.collect()
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()
                break
        for i in range(num_devices):
            self.caching_allocator_befores.append(torch.cuda.memory_allocated(i))
            bytes_free, bytes_total = torch.cuda.mem_get_info(i)
            driver_mem_allocated = bytes_total - bytes_free
            self.driver_befores.append(driver_mem_allocated)

    def __exit__(self, exec_type, exec_value, traceback):
        if exec_type is not None:
            return
        discrepancy_detected = False
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            torch._C._cuda_clearCublasWorkspaces()
            caching_allocator_mem_allocated = torch.cuda.memory_allocated(i)
            if caching_allocator_mem_allocated > self.caching_allocator_befores[i]:
                discrepancy_detected = True
                break
        if not discrepancy_detected:
            return
        gc.collect()
        torch.cuda.empty_cache()
        for i in range(num_devices):
            discrepancy_detected = True
            for n in range(3):
                caching_allocator_mem_allocated = torch.cuda.memory_allocated(i)
                bytes_free, bytes_total = torch.cuda.mem_get_info(i)
                driver_mem_allocated = bytes_total - bytes_free
                caching_allocator_discrepancy = False
                driver_discrepancy = False
                if caching_allocator_mem_allocated > self.caching_allocator_befores[i]:
                    caching_allocator_discrepancy = True
                if driver_mem_allocated > self.driver_befores[i]:
                    driver_discrepancy = True
                if not (caching_allocator_discrepancy or driver_discrepancy):
                    discrepancy_detected = False
                    break
            if not discrepancy_detected:
                continue
            if caching_allocator_discrepancy and (not driver_discrepancy):
                msg = 'CUDA caching allocator reports a memory leak not verified by the driver API in {}! Caching allocator allocated memory was {} and is now reported as {} on device {}. CUDA driver allocated memory was {} and is now {}.'.format(self.name, self.caching_allocator_befores[i], caching_allocator_mem_allocated, i, self.driver_befores[i], driver_mem_allocated)
                warnings.warn(msg)
            elif caching_allocator_discrepancy and driver_discrepancy:
                msg = 'CUDA driver API confirmed a leak in {}! Caching allocator allocated memory was {} and is now reported as {} on device {}. CUDA driver allocated memory was {} and is now {}.'.format(self.name, self.caching_allocator_befores[i], caching_allocator_mem_allocated, i, self.driver_befores[i], driver_mem_allocated)
                raise RuntimeError(msg)