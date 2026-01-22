import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import itertools
import abc
import _thread
from types import FunctionType, GenericAlias
def _is_initvar(a_type, dataclasses):
    return a_type is dataclasses.InitVar or type(a_type) is dataclasses.InitVar