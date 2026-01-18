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
@functools.wraps(func)
def wrapped_no_args(self):
    if not hasattr(self, cache_name):
        object.__setattr__(self, cache_name, func(self))
    return getattr(self, cache_name)