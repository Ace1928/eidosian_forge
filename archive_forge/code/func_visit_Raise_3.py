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
def visit_Raise_3(self, node):
    if node.exc:
        self.attr(node, 'open_raise', ['raise', self.ws], default='raise ')
        self.visit(node.exc)
        if node.cause:
            self.attr(node, 'cause_prefix', [self.ws, 'from', self.ws], default=' from ')
            self.visit(node.cause)
    else:
        self.token('raise')