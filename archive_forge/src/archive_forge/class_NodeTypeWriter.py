from __future__ import absolute_import
import os
import re
import unittest
import shlex
import sys
import tempfile
import textwrap
from io import open
from functools import partial
from .Compiler import Errors
from .CodeWriter import CodeWriter
from .Compiler.TreeFragment import TreeFragment, strip_common_indent
from .Compiler.Visitor import TreeVisitor, VisitorTransform
from .Compiler import TreePath
class NodeTypeWriter(TreeVisitor):

    def __init__(self):
        super(NodeTypeWriter, self).__init__()
        self._indents = 0
        self.result = []

    def visit_Node(self, node):
        if not self.access_path:
            name = u'(root)'
        else:
            tip = self.access_path[-1]
            if tip[2] is not None:
                name = u'%s[%d]' % tip[1:3]
            else:
                name = tip[1]
        self.result.append(u'  ' * self._indents + u'%s: %s' % (name, node.__class__.__name__))
        self._indents += 1
        self.visitchildren(node)
        self._indents -= 1