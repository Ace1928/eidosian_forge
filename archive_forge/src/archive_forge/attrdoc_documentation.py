import ast
import inspect
import sys
import textwrap
import typing as T
from types import ModuleType
from .common import Docstring, DocstringParam
Get attribute docstrings from the given component.

        :param component: component to process (class or module)
        :returns: for each attribute docstring, a tuple with (description,
            type, default)
        