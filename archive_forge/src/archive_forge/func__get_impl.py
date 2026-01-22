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
def _get_impl(self, args, kws):
    """Get implementation given the argument types.

        Returning a Dispatcher object.  The Dispatcher object is cached
        internally in `self._impl_cache`.
        """
    flags = targetconfig.ConfigStack.top_or_none()
    cache_key = (self.context, tuple(args), tuple(kws.items()), flags)
    try:
        impl, args = self._impl_cache[cache_key]
        return (impl, args)
    except KeyError:
        pass
    impl, args = self._build_impl(cache_key, args, kws)
    return (impl, args)