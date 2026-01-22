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
def _build_impl(self, cache_key, args, kws):
    """Build and cache the implementation.

        Given the positional (`args`) and keyword arguments (`kws`), obtains
        the `overload` implementation and wrap it in a Dispatcher object.
        The expected argument types are returned for use by type-inference.
        The expected argument types are only different from the given argument
        types if there is an imprecise type in the given argument types.

        Parameters
        ----------
        cache_key : hashable
            The key used for caching the implementation.
        args : Tuple[Type]
            Types of positional argument.
        kws : Dict[Type]
            Types of keyword argument.

        Returns
        -------
        disp, args :
            On success, returns `(Dispatcher, Tuple[Type])`.
            On failure, returns `(None, None)`.

        """
    jitter = self._get_jit_decorator()
    ov_sig = inspect.signature(self._overload_func)
    try:
        ov_sig.bind(*args, **kws)
    except TypeError as e:
        raise TypingError(str(e)) from e
    else:
        ovf_result = self._overload_func(*args, **kws)
    if ovf_result is None:
        self._impl_cache[cache_key] = (None, None)
        return (None, None)
    elif isinstance(ovf_result, tuple):
        sig, pyfunc = ovf_result
        args = sig.args
        kws = {}
        cache_key = None
    else:
        pyfunc = ovf_result
    if not isinstance(pyfunc, FunctionType):
        msg = 'Implementation function returned by `@overload` has an unexpected type.  Got {}'
        raise AssertionError(msg.format(pyfunc))
    if self._strict:
        self._validate_sigs(self._overload_func, pyfunc)
    jitdecor = jitter(**self._jit_options)
    disp = jitdecor(pyfunc)
    disp_type = types.Dispatcher(disp)
    disp_type.get_call_type(self.context, args, kws)
    if cache_key is not None:
        self._impl_cache[cache_key] = (disp, args)
    return (disp, args)