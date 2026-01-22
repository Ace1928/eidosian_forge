import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _keyword(self, t):
    if t.arg:
        self.write(t.arg)
    else:
        self.write('**')
    self.write('=')
    self.dispatch(t.value)