import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def _loop_reference_detected(self, node):
    if 'loop' in node.undeclared_identifiers():
        self.detected = True
    else:
        for n in node.get_children():
            n.accept_visitor(self)