import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def visitTextTag(self, node):
    for ident in node.undeclared_identifiers():
        if ident != 'context' and ident not in self.declared.union(self.locally_declared):
            self.undeclared.add(ident)