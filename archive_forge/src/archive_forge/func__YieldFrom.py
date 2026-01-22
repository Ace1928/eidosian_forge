import ast
import io
import sys
import tokenize
def _YieldFrom(self, t):
    self.write('(')
    self.write('yield from')
    if t.value:
        self.write(' ')
        self.dispatch(t.value)
    self.write(')')