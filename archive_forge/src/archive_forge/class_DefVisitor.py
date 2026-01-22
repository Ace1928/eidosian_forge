import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
class DefVisitor:

    def visitDefTag(s, node):
        s.visitDefOrBase(node)

    def visitBlockTag(s, node):
        s.visitDefOrBase(node)

    def visitDefOrBase(s, node):
        self.write_inline_def(node, callable_identifiers, nested=False)
        if not node.is_anonymous:
            export.append(node.funcname)
        if node.funcname in body_identifiers.closuredefs:
            del body_identifiers.closuredefs[node.funcname]