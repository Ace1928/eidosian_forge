import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def write_def_decl(self, node, identifiers):
    """write a locally-available callable referencing a top-level def"""
    funcname = node.funcname
    namedecls = node.get_argument_expressions()
    nameargs = node.get_argument_expressions(as_call=True)
    if not self.in_def and (len(self.identifiers.locally_assigned) > 0 or len(self.identifiers.argument_declared) > 0):
        nameargs.insert(0, 'context._locals(__M_locals)')
    else:
        nameargs.insert(0, 'context')
    self.printer.writeline('def %s(%s):' % (funcname, ','.join(namedecls)))
    self.printer.writeline('return render_%s(%s)' % (funcname, ','.join(nameargs)))
    self.printer.writeline(None)