import sys
import traceback
from bisect import bisect_left, bisect_right, insort
from itertools import chain, repeat, starmap
from math import log
from operator import add, eq, ne, gt, ge, lt, le, iadd
from textwrap import dedent
from functools import wraps
from sys import hexversion
def __make_cmp(seq_op, symbol, doc):
    """Make comparator method."""

    def comparer(self, other):
        """Compare method for sorted list and sequence."""
        if not isinstance(other, Sequence):
            return NotImplemented
        self_len = self._len
        len_other = len(other)
        if self_len != len_other:
            if seq_op is eq:
                return False
            if seq_op is ne:
                return True
        for alpha, beta in zip(self, other):
            if alpha != beta:
                return seq_op(alpha, beta)
        return seq_op(self_len, len_other)
    seq_op_name = seq_op.__name__
    comparer.__name__ = '__{0}__'.format(seq_op_name)
    doc_str = 'Return true if and only if sorted list is {0} `other`.\n\n        ``sl.__{1}__(other)`` <==> ``sl {2} other``\n\n        Comparisons use lexicographical order as with sequences.\n\n        Runtime complexity: `O(n)`\n\n        :param other: `other` sequence\n        :return: true if sorted list is {0} `other`\n\n        '
    comparer.__doc__ = dedent(doc_str.format(doc, seq_op_name, symbol))
    return comparer