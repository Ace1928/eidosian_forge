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
def _wrap_load_module(self, method: Any) -> Any:

    def load_module(fullname: str) -> ModuleType:
        assert fullname == self.old_module_name, f'DeprecatedModuleLoader for {self.old_module_name} was asked to load {fullname}'
        if self.new_module_name in sys.modules:
            sys.modules[self.old_module_name] = sys.modules[self.new_module_name]
            return sys.modules[self.old_module_name]
        method(self.new_module_name)
        assert self.new_module_name in sys.modules, f'Wrapped loader {self.loader} was expected to insert {self.new_module_name} in sys.modules but it did not.'
        sys.modules[self.old_module_name] = sys.modules[self.new_module_name]
        return sys.modules[self.old_module_name]
    return load_module