from numba.core import types, errors, cgutils
from numba.core.extending import intrinsic
Basically do ``isinstance(exc_value, exc_class)`` for exception objects.
    Used in ``except Exception:`` syntax.
    