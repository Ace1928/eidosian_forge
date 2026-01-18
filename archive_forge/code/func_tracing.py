from __future__ import annotations
import contextlib
import dataclasses
import enum
import functools
import logging
import threading
import traceback
import unittest.mock
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
import torch
from torch.utils import _pytree as pytree
from torch.utils._traceback import CapturedTraceback
from torch.utils.weak import WeakTensorKeyDictionary
@contextmanager
def tracing(context: Optional[TracingContext]):
    """
    This function installs the passed in tracing context as a dynamic scoped
    global variable.

    Calls to TracingContext.get() while not under a `with tracing()` context
    will return None.
    """
    old_context = getattr(_TLS, 'tracing_context', None)
    _TLS.tracing_context = context
    try:
        yield context
    except Exception as e:
        if not hasattr(e, 'real_stack') and context is not None:
            e.real_stack = context.extract_stack()
        raise
    finally:
        if context is not None and context.fake_mode is not None and (context.fake_mode.shape_env is not None):
            context.fake_mode.shape_env.cleanup()
        _TLS.tracing_context = old_context