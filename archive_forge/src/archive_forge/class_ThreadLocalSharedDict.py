from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
class ThreadLocalSharedDict(threading.local):
    """
    Descriptor that holds a dict shared between instances of a class in the same thread.

    Note: Descriptors have slightly different semantics than just a dict field on its own.
    `PartialState(...)._shared_state` and `PartialState._shared_state` (instance vs class) give the same value: the
    underlying _storage dict. Likewise, `PartialState(...)._shared_state = {...}` overrides the _storage dict inside
    the descriptor as you would expect. However, `PartialState._shared_state = {}` actually replaces the descriptor
    object with a dict instead Thus, you should modify the _storage dict in-place (e.g. `_shared_state.clear()`).

    See Python documentation for an explanation of descriptors: https://docs.python.org/3/howto/descriptor.html

    This is required for using PyTorch/XLA with PJRT in multithreaded mode (required for TPU v2 and v3).

    See https://github.com/pytorch/xla/blob/r2.0/docs/pjrt.md#multithreading-on-tpu-v2v3
    """

    def __init__(self, thread_local: bool=False):
        self._storage = {}

    def __get__(self, obj, objtype=None):
        return self._storage

    def __set__(self, obj, value):
        self._storage = value