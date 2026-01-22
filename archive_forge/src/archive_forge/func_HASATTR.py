from __future__ import annotations
import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import logging
import math
import os
import re
import sys
import textwrap
import types
import weakref
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from weakref import ReferenceType
import torch
import torch.utils._device
from torch._dynamo.source import (
from torch._guards import (
from torch.fx.experimental.symbolic_shapes import (
from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef
from . import config, convert_frame, exc, mutation_guard
from .eval_frame import set_guard_error_hook
from .source import DefaultsSource, LocalSource, TypeSource
from .types import GuardedCode, GuardFail, GuardFn  # noqa: F401
from .utils import (
def HASATTR(self, guard: Guard):
    m = re.match('^(.*)[.]([a-zA-Z0-9_]+)$', guard.name)
    assert m, f'invalid hasattr check {guard.name}'
    base, attr = m.group(1, 2)
    ref = self.arg_ref(base)
    val = hasattr(self.get(base), attr)
    code = None
    if val:
        code = f'hasattr({ref}, {attr!r})'
    else:
        code = f'not hasattr({ref}, {attr!r})'
    self._produce_guard_code(guard, [code], provided_guarded_object=self.get(base))