import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _If(self, t):
    self.fill('if ')
    self.dispatch(t.test)
    self.enter()
    self.dispatch(t.body)
    self.leave()
    while t.orelse and len(t.orelse) == 1 and isinstance(t.orelse[0], ast.If):
        t = t.orelse[0]
        self.fill('elif ')
        self.dispatch(t.test)
        self.enter()
        self.dispatch(t.body)
        self.leave()
    if t.orelse:
        self.fill('else')
        self.enter()
        self.dispatch(t.orelse)
        self.leave()