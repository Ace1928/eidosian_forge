from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def unify_types(self, *typelist):

    def keyfunc(obj):
        """Uses bitwidth to order numeric-types.
            Fallback to stable, deterministic sort.
            """
        return getattr(obj, 'bitwidth', 0)
    typelist = sorted(typelist, key=keyfunc)
    unified = typelist[0]
    for tp in typelist[1:]:
        unified = self.unify_pairs(unified, tp)
        if unified is None:
            break
    return unified