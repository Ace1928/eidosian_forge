import atexit
import collections
import contextlib
import copy
import cProfile
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import inspect
import itertools
import linecache
import logging
import math
import operator
import os
import pstats
import subprocess
import sys
import textwrap
import threading
import time
import types
import typing
import weakref
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
import importlib
import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._pytree import tree_map_only
from torch._subclasses import (  # noqa: F401
def to_numpy_helper(value):
    """Convert tensor and tnp.ndarray to numpy.ndarray."""
    if isinstance(value, tnp.ndarray):
        return to_numpy_helper(value.tensor)
    elif isinstance(value, torch.Tensor):
        return value.numpy(force=True)
    elif isinstance(value, (tuple, list)):
        return type(value)((to_numpy_helper(obj) for obj in value))
    else:
        return value