import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Compare(self, t):
    self.write('(')
    self.dispatch(t.left)
    for o, e in zip(t.ops, t.comparators):
        self.write(' ' + self.cmpops[o.__class__.__name__] + ' ')
        self.dispatch(e)
    self.write(')')