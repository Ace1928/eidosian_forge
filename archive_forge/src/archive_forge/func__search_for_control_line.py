import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def _search_for_control_line():
    for c in children:
        if isinstance(c, parsetree.Comment):
            continue
        elif isinstance(c, parsetree.ControlLine):
            return True
        return False