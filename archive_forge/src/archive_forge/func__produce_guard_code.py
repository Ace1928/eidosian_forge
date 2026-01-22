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
def _produce_guard_code(self, guard, code_list, provided_guarded_object=None, shape_env=False):
    cur_frame = currentframe()
    assert cur_frame is not None
    caller = cur_frame.f_back
    del cur_frame
    assert caller is not None
    func_name = getframeinfo(caller)[2]
    del caller
    assert func_name in dir(self.__class__), f'_produce_guard_code must be called from inside GuardedCode. Called from {func_name}'
    if shape_env:
        self.shape_env_code.append(GuardCodeList(code_list, guard))
    else:
        self.code.append(GuardCodeList(code_list, guard))
    if provided_guarded_object is None:
        name_valid = guard.name is not None and guard.name != ''
        guarded_object = self.get(guard.name) if name_valid else None
    else:
        guarded_object = provided_guarded_object
    guarded_object_type = weakref.ref(type(guarded_object)) if guarded_object is not None else None
    obj_ref = None
    if hasattr(guarded_object.__class__, '__weakref__') and (not isinstance(guarded_object, enum.Enum)):
        obj_ref = weakref.ref(guarded_object)
    guard.set_export_info(func_name, guarded_object_type, code_list, obj_ref)