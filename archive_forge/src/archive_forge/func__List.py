import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _List(self, t):
    self.write('[')
    interleave(lambda: self.write(', '), self.dispatch, t.elts)
    self.write(']')