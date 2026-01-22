import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Dict(self, t):
    self.write('{')

    def write_pair(pair):
        k, v = pair
        self.dispatch(k)
        self.write(': ')
        self.dispatch(v)
    interleave(lambda: self.write(', '), write_pair, zip(t.keys, t.values))
    self.write('}')