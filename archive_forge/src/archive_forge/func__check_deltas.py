from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
def _check_deltas(self, growth):
    deltas = self.deltas
    if not deltas:
        return True
    if gc.garbage:
        raise LeakCheckError('Generated uncollectable garbage %r' % (gc.garbage,))
    if deltas[-2:] == [0, 0] and len(deltas) in (2, 3):
        return False
    if deltas[-3:] == [0, 0, 0]:
        return False
    if len(deltas) >= 4 and sum(deltas[-4:]) == 0:
        return False
    if len(deltas) >= 3 and deltas[-1] > 0 and (deltas[-1] == deltas[-2]) and (deltas[-2] == deltas[-3]):
        diff = self._report_diff(growth)
        raise LeakCheckError('refcount increased by %r\n%s' % (deltas, diff))
    if sum(deltas[-3:]) <= 0 or sum(deltas[-4:]) <= 0 or deltas[-4:].count(0) >= 2:
        limit = 11
    else:
        limit = 7
    if len(deltas) >= limit:
        raise LeakCheckError('refcount increased by %r\n%s' % (deltas, self._report_diff(growth)))
    return True