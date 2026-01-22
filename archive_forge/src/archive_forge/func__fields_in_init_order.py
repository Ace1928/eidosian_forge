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
def _fields_in_init_order(fields):
    return (tuple((f for f in fields if f.init and (not f.kw_only))), tuple((f for f in fields if f.init and f.kw_only)))