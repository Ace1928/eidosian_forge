from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class SeqSortRef(SortRef):
    """Sequence sort."""

    def is_string(self):
        """Determine if sort is a string
        >>> s = StringSort()
        >>> s.is_string()
        True
        >>> s = SeqSort(IntSort())
        >>> s.is_string()
        False
        """
        return Z3_is_string_sort(self.ctx_ref(), self.ast)

    def basis(self):
        return _to_sort_ref(Z3_get_seq_sort_basis(self.ctx_ref(), self.ast), self.ctx)