import functools
import inspect
import textwrap
import threading
import types
import warnings
from typing import TypeVar, Type, Callable, Any, Union
from inspect import signature
from functools import wraps
class lazyclassproperty(classproperty):

    def __new__(cls, fget=None, doc=None):
        return super().__new__(cls, fget, doc, lazy=True)