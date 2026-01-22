from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import re
from pasta.augment import errors
from pasta.base import formatting as fmt
class FindNodeVisitor(ast.NodeVisitor):

    def __init__(self, condition):
        self._condition = condition
        self.results = []

    def visit(self, node):
        if self._condition(node):
            self.results.append(node)
        super(FindNodeVisitor, self).visit(node)