import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def _decorate_getattr(impl, ty, attr):
    real_impl = impl
    if attr is not None:

        def res(context, builder, typ, value, attr):
            return real_impl(context, builder, typ, value)
    else:

        def res(context, builder, typ, value, attr):
            return real_impl(context, builder, typ, value, attr)
    res.signature = (ty,)
    res.attr = attr
    return res