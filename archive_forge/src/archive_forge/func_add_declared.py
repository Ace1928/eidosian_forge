import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def add_declared(self, ident):
    self.declared.add(ident)
    if ident in self.undeclared:
        self.undeclared.remove(ident)