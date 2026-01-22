import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _ImportFrom(self, t):
    if t.module and t.module == '__future__':
        self.future_imports.extend((n.name for n in t.names))
    self.fill('from ')
    self.write('.' * t.level)
    if t.module:
        self.write(t.module)
    self.write(' import ')
    interleave(lambda: self.write(', '), self.dispatch, t.names)