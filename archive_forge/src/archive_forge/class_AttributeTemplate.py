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
class AttributeTemplate(object):

    def __init__(self, context):
        self.context = context

    def resolve(self, value, attr):
        return self._resolve(value, attr)

    def _resolve(self, value, attr):
        fn = getattr(self, 'resolve_%s' % attr, None)
        if fn is None:
            fn = self.generic_resolve
            if fn is NotImplemented:
                if isinstance(value, types.Module):
                    return self.context.resolve_module_constants(value, attr)
                else:
                    return None
            else:
                return fn(value, attr)
        else:
            return fn(value)
    generic_resolve = NotImplemented