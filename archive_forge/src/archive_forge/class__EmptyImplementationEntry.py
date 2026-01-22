from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
class _EmptyImplementationEntry(InternalError):

    def __init__(self, reason):
        super(_EmptyImplementationEntry, self).__init__('_EmptyImplementationEntry({!r})'.format(reason))