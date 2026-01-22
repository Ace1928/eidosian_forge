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
def RAISE(self, node):
    self.handleChildren(node)
    arg = node.exc
    if isinstance(arg, ast.Call):
        if is_notimplemented_name_node(arg.func):
            self.report(messages.RaiseNotImplemented, node)
    elif is_notimplemented_name_node(arg):
        self.report(messages.RaiseNotImplemented, node)