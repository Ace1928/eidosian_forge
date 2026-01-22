from __future__ import annotations
from typing import Any
import builtins
import inspect
import keyword
import textwrap
import linecache
from sympy.external import import_module # noqa:F401
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import (is_sequence, iterable,
from sympy.utilities.misc import filldedent
def _print_unpacking(self, lvalues, rvalue):
    """Generate argument unpacking code.

        This method is used when the input value is not interable,
        but can be indexed (see issue #14655).
        """

    def flat_indexes(elems):
        n = 0
        for el in elems:
            if iterable(el):
                for ndeep in flat_indexes(el):
                    yield ((n,) + ndeep)
            else:
                yield (n,)
            n += 1
    indexed = ', '.join(('{}[{}]'.format(rvalue, ']['.join(map(str, ind))) for ind in flat_indexes(lvalues)))
    return ['[{}] = [{}]'.format(', '.join(flatten(lvalues)), indexed)]