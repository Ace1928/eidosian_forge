import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
def assertNoCode(inst, attribute, f):
    if f.__code__.co_code not in (_empty.__code__.co_code, _docstring.__code__.co_code):
        raise ValueError('function body must be empty')