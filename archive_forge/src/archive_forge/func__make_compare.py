import operator as op
from collections import OrderedDict
from collections.abc import (
from contextlib import contextmanager
from shutil import rmtree
from .core import ENOVAL, Cache
def _make_compare(seq_op, doc):
    """Make compare method with Sequence semantics."""

    def compare(self, that):
        """Compare method for deque and sequence."""
        if not isinstance(that, Sequence):
            return NotImplemented
        len_self = len(self)
        len_that = len(that)
        if len_self != len_that:
            if seq_op is op.eq:
                return False
            if seq_op is op.ne:
                return True
        for alpha, beta in zip(self, that):
            if alpha != beta:
                return seq_op(alpha, beta)
        return seq_op(len_self, len_that)
    compare.__name__ = '__{0}__'.format(seq_op.__name__)
    doc_str = 'Return True if and only if deque is {0} `that`.'
    compare.__doc__ = doc_str.format(doc)
    return compare