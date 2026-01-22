import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Attribute(self, t):
    self.dispatch(t.value)
    if isnum(t.value) and isinstance(t.value.value, int):
        self.write(' ')
    self.write('.')
    self.write(t.attr)