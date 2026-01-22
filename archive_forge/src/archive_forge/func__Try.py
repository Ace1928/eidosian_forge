import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Try(self, t):
    self.fill('try')
    self.enter()
    self.dispatch(t.body)
    self.leave()
    for ex in t.handlers:
        self.dispatch(ex)
    if t.orelse:
        self.fill('else')
        self.enter()
        self.dispatch(t.orelse)
        self.leave()