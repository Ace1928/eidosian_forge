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
def DICT_KEYS(self, guard):
    ref = self.arg_ref(guard)
    value = self.get(guard.name)
    t = type(value)
    code = list()
    code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
    param_key_ids = set(dict_param_key_ids(value))
    const_keys = set(dict_const_keys(value))
    const_keys_repr = dict_const_keys_repr(const_keys, local=is_from_local_source(guard.originating_source))
    if param_key_ids:
        code.append(f'___dict_param_key_ids({ref}) == {param_key_ids!r}')
        code.append(f'___dict_const_keys({ref}) == {const_keys_repr}')
    else:
        code.append(f'set({ref}.keys()) == {const_keys_repr}')
    self._produce_guard_code(guard, code)