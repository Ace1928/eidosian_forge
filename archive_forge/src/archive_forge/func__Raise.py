import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Raise(self, t):
    self.fill('raise ')
    if t.exc:
        self.dispatch(t.exc)
    if t.cause:
        self.write('from ')
        self.dispatch(t.cause)