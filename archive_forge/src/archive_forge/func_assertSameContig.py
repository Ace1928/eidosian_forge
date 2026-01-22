import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def assertSameContig(self, arr, nparr):
    attrs = ('C_CONTIGUOUS', 'F_CONTIGUOUS')
    for attr in attrs:
        if arr.flags[attr] != nparr.flags[attr]:
            if arr.size == 0 and nparr.size == 0:
                pass
            else:
                self.fail('contiguous flag mismatch:\ngot=%s\nexpect=%s' % (arr.flags, nparr.flags))