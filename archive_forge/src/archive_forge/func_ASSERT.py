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
def ASSERT(self, node):
    if isinstance(node.test, ast.Tuple) and node.test.elts != []:
        self.report(messages.AssertTuple, node)
    self.handleChildren(node)