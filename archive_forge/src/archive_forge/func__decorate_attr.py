import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def _decorate_attr(self, impl, ty, attr, impl_list, decorator):
    real_impl = decorator(impl, ty, attr)
    impl_list.append((real_impl, attr, real_impl.signature))
    return impl