import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Str(self, tree):
    if 'unicode_literals' not in self.future_imports:
        self.write(repr(tree.value))
    elif isinstance(tree.value, str):
        self.write('b' + repr(tree.value))
    else:
        assert False, "shouldn't get here"