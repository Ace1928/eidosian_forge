import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
class FindTopLevel:

    def visitInheritTag(s, node):
        inherit.append(node)

    def visitNamespaceTag(s, node):
        namespaces[node.name] = node

    def visitPageTag(s, node):
        self.compiler.pagetag = node

    def visitCode(s, node):
        if node.ismodule:
            module_code.append(node)