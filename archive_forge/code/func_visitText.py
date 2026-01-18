import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def visitText(self, node):
    self.printer.start_source(node.lineno)
    self.printer.writeline('__M_writer(%s)' % repr(node.content))