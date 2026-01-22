import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _DictComp(self, t):
    self.write('{')
    self.dispatch(t.key)
    self.write(': ')
    self.dispatch(t.value)
    for gen in t.generators:
        self.dispatch(gen)
    self.write('}')