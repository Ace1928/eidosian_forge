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
def TRY(self, node):
    handler_names = []
    for i, handler in enumerate(node.handlers):
        if isinstance(handler.type, ast.Tuple):
            for exc_type in handler.type.elts:
                handler_names.append(getNodeName(exc_type))
        elif handler.type:
            handler_names.append(getNodeName(handler.type))
        if handler.type is None and i < len(node.handlers) - 1:
            self.report(messages.DefaultExceptNotLast, handler)
    self.exceptHandlers.append(handler_names)
    for child in node.body:
        self.handleNode(child, node)
    self.exceptHandlers.pop()
    self.handleChildren(node, omit='body')