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
class _BrokenModule(ModuleType):

    def __init__(self, name, exc):
        self.exc = exc
        super().__init__(name)

    def __getattr__(self, name):
        raise self.exc