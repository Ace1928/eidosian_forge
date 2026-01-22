import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
class NSDefVisitor:

    def visitDefTag(s, node):
        s.visitDefOrBase(node)

    def visitBlockTag(s, node):
        s.visitDefOrBase(node)

    def visitDefOrBase(s, node):
        if node.is_anonymous:
            raise exceptions.CompileException("Can't put anonymous blocks inside <%namespace>", **node.exception_kwargs)
        self.write_inline_def(node, identifiers, nested=False)
        export.append(node.funcname)