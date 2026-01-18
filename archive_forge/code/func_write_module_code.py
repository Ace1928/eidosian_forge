import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def write_module_code(self, module_code):
    """write module-level template code, i.e. that which
        is enclosed in <%! %> tags in the template."""
    for n in module_code:
        self.printer.write_indented_block(n.text, starting_lineno=n.lineno)