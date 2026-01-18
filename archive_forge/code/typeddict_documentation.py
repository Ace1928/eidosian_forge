from collections.abc import MutableMapping, Iterable, Mapping
from numba.core.types import DictType
from numba.core.imputils import numba_typeref_ctor
from numba import njit, typeof
from numba.core import types, errors, config, cgutils
from numba.core.extending import (
from numba.typed import dictobject
from numba.core.typing import signature
Returns True if the dictionary is typed.
        