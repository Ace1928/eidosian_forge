import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Delete(self, t):
    self.fill('del ')
    interleave(lambda: self.write(', '), self.dispatch, t.targets)