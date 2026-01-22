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
def FUNCTIONDEF(self, node):
    for deco in node.decorator_list:
        self.handleNode(deco, node)
    with self._type_param_scope(node):
        self.LAMBDA(node)
    self.addBinding(node, FunctionDefinition(node.name, node))
    if self.withDoctest and (not self._in_doctest()) and (not isinstance(self.scope, FunctionScope)):
        self.deferFunction(lambda: self.handleDoctests(node))