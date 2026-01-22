import ast
import io
import sys
import tokenize
def _AsyncWith(self, t):
    self.fill('async with ')
    interleave(lambda: self.write(', '), self.dispatch, t.items)
    self.enter()
    self.dispatch(t.body)
    self.leave()