import contextlib
import contextvars
import dataclasses
import functools
import importlib
import inspect
import os
import re
import sys
import traceback
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Optional, overload, Set, Tuple, Type, TypeVar
import numpy as np
import pandas as pd
import sympy
import sympy.printing.repr
from cirq._doc import document
def _setup_deprecated_submodule_attribute(new_module_name: str, old_parent: str, old_child: str, deadline: str, new_module: Optional[ModuleType]):
    parent_module = sys.modules[old_parent]
    setattr(parent_module, old_child, new_module)

    class Wrapped(ModuleType):
        __dict__ = parent_module.__dict__
        __spec__ = _make_proxy_spec_property(parent_module)

        def __getattr__(self, name):
            if name == old_child:
                _deduped_module_warn_or_error(f'{old_parent}.{old_child}', new_module_name, deadline)
            return getattr(parent_module, name)
    wrapped_parent_module = Wrapped(parent_module.__name__, parent_module.__doc__)
    if '.' in old_parent:
        grandpa_name, parent_tail = old_parent.rsplit('.', 1)
        grandpa_module = sys.modules[grandpa_name]
        setattr(grandpa_module, parent_tail, wrapped_parent_module)
    sys.modules[old_parent] = wrapped_parent_module