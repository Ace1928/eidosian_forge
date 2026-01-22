import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def CONTINUE(self, node):
    n = node
    while hasattr(n, '_pyflakes_parent'):
        n, n_child = (n._pyflakes_parent, n)
        if isinstance(n, (ast.While, ast.For, ast.AsyncFor)):
            if n_child not in n.orelse:
                return
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            break
    if isinstance(node, ast.Continue):
        self.report(messages.ContinueOutsideLoop, node)
    else:
        self.report(messages.BreakOutsideLoop, node)