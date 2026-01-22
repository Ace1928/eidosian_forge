import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
def _hash_basis(self):
    params = tuple((param for param in self.parameters.values() if param.kind != _KEYWORD_ONLY))
    kwo_params = {param.name: param for param in self.parameters.values() if param.kind == _KEYWORD_ONLY}
    return (params, kwo_params, self.return_annotation)