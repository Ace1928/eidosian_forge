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
def unify_pairs(self, first, second):
    """
        Try to unify the two given types.  A third type is returned,
        or None in case of failure.
        """
    if first == second:
        return first
    if first is types.undefined:
        return second
    elif second is types.undefined:
        return first
    unified = first.unify(self, second)
    if unified is not None:
        return unified
    unified = second.unify(self, first)
    if unified is not None:
        return unified
    conv = self.can_convert(fromty=first, toty=second)
    if conv is not None and conv <= Conversion.safe:
        return second
    conv = self.can_convert(fromty=second, toty=first)
    if conv is not None and conv <= Conversion.safe:
        return first
    if isinstance(first, types.Literal) or isinstance(second, types.Literal):
        first = types.unliteral(first)
        second = types.unliteral(second)
        return self.unify_pairs(first, second)
    return None