import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _With(self, t):
    self.fill('with ')
    self.dispatch(t.context_expr)
    if t.optional_vars:
        self.write(' as ')
        self.dispatch(t.optional_vars)
    self.enter()
    self.dispatch(t.body)
    self.leave()