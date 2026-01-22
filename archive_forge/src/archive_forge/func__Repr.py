import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Repr(self, t):
    self.write('`')
    self.dispatch(t.value)
    self.write('`')