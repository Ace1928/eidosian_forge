from __future__ import annotations
import sys
from typing import Any, Type
import inspect
from contextlib import contextmanager
from functools import cmp_to_key, update_wrapper
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import AppliedUndef, UndefinedFunction, Function
@classmethod
def _get_initial_settings(cls):
    settings = cls._default_settings.copy()
    for key, val in cls._global_settings.items():
        if key in cls._default_settings:
            settings[key] = val
    return settings