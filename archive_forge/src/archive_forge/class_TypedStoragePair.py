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
class TypedStoragePair(TensorLikePair):
    """Pair for :class:`torch.storage.TypedStorage` inputs."""

    def __init__(self, actual, expected, *, rtol_override=0.0, atol_override=0.0, **other_parameters):
        self._check_inputs_isinstance(actual, expected, cls=torch.storage.TypedStorage)
        super().__init__(actual, expected, **other_parameters)
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    def _to_tensor(self, typed_storage):
        return torch.tensor(typed_storage._untyped_storage, dtype={torch.quint8: torch.uint8, torch.quint4x2: torch.uint8, torch.quint2x4: torch.uint8, torch.qint32: torch.int32, torch.qint8: torch.int8}.get(typed_storage.dtype, typed_storage.dtype), device=typed_storage.device)