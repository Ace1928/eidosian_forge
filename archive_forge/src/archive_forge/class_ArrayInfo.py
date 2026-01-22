import contextlib
import functools
import inspect
import os
import platform
import random
import tempfile
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
import numpy
from packaging.version import Version
from wasabi import table
from .compat import (
from .compat import mxnet as mx
from .compat import tensorflow as tf
from .compat import torch
from typing import TYPE_CHECKING
from . import types  # noqa: E402
from .types import ArgsKwargs, ArrayXd, FloatsXd, IntsXd, Padded, Ragged  # noqa: E402
@dataclass
class ArrayInfo:
    """Container for info for checking array compatibility."""
    shape: types.Shape
    dtype: types.DTypes

    @classmethod
    def from_array(cls, arr: ArrayXd):
        return cls(shape=arr.shape, dtype=arr.dtype)

    def check_consistency(self, arr: ArrayXd):
        if arr.shape != self.shape:
            raise ValueError(f'Shape mismatch in backprop. Y: {self.shape}, dY: {arr.shape}')
        if arr.dtype != self.dtype:
            raise ValueError(f'Type mismatch in backprop. Y: {self.dtype}, dY: {arr.dtype}')