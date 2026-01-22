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
def _to_number(self, number_like, *, id):
    if isinstance(number_like, (torch.Tensor, np.ndarray)):
        numel = number_like.numel() if isinstance(number_like, torch.Tensor) else number_like.size
        if numel > 1:
            self._fail(ValueError, f'Only single element tensor-likes can be compared against a number. Got {numel} elements instead.', id=id)
        number = number_like.item()
        if isinstance(number, bool):
            number = int(number)
        return number
    elif isinstance(number_like, Enum):
        return int(number_like)
    else:
        return super()._to_number(number_like, id=id)