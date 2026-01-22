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
def assertGreaterAlmostEqual(self, first, second, places=None, msg=None, delta=None):
    """Assert that ``first`` is greater than or almost equal to ``second``.

        The equality of ``first`` and ``second`` is determined in a similar way to
        the ``assertAlmostEqual`` function of the standard library.
        """
    if delta is not None and places is not None:
        raise TypeError('specify delta or places not both')
    if first >= second:
        return
    diff = second - first
    if delta is not None:
        if diff <= delta:
            return
        standardMsg = f'{first} not greater than or equal to {second} within {delta} delta'
    else:
        if places is None:
            places = 7
        if round(diff, places) == 0:
            return
        standardMsg = f'{first} not greater than or equal to {second} within {places} places'
    msg = self._formatMessage(msg, standardMsg)
    raise self.failureException(msg)