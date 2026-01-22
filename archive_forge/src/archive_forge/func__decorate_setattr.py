import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def _decorate_setattr(impl, ty, attr):
    real_impl = impl
    if attr is not None:

        def res(context, builder, sig, args, attr):
            return real_impl(context, builder, sig, args)
    else:

        def res(context, builder, sig, args, attr):
            return real_impl(context, builder, sig, args, attr)
    res.signature = (ty, types.Any)
    res.attr = attr
    return res