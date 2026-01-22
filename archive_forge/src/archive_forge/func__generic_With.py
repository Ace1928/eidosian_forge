from __future__ import print_function, unicode_literals
import six
import sys
import ast
import os
import tokenize
from six import StringIO
def _generic_With(self, t, async_=False):
    self.fill('async with ' if async_ else 'with ')
    if hasattr(t, 'items'):
        interleave(lambda: self.write(', '), self.dispatch, t.items)
    else:
        self.dispatch(t.context_expr)
        if t.optional_vars:
            self.write(' as ')
            self.dispatch(t.optional_vars)
    self.enter()
    self.dispatch(t.body)
    self.leave()