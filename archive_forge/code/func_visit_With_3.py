from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import ast
import contextlib
import functools
import itertools
import six
from six.moves import zip
import sys
from pasta.base import ast_constants
from pasta.base import ast_utils
from pasta.base import formatting as fmt
from pasta.base import token_generator
def visit_With_3(self, node):
    if hasattr(ast, 'AsyncWith') and isinstance(node, ast.AsyncWith):
        self.attr(node, 'with', ['async', self.ws, 'with', self.ws], default='async with ')
    else:
        self.attr(node, 'with', ['with', self.ws], default='with ')
    for i, withitem in enumerate(node.items):
        self.visit(withitem)
        if i != len(node.items) - 1:
            self.token(',')
    self.attr(node, 'with_body_open', [':', self.ws_oneline], default=':\n')
    for stmt in self.indented(node, 'body'):
        self.visit(stmt)