from collections.abc import MutableMapping, Iterable, Mapping
from numba.core.types import DictType
from numba.core.imputils import numba_typeref_ctor
from numba import njit, typeof
from numba.core import types, errors, config, cgutils
from numba.core.extending import (
from numba.typed import dictobject
from numba.core.typing import signature
def _initialise_dict(self, key, value):
    dcttype = types.DictType(typeof(key), typeof(value))
    self._dict_type, self._opaque = self._parse_arg(dcttype)