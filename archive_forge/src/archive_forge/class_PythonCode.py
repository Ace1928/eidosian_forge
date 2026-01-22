import re
from mako import exceptions
from mako import pyparser
class PythonCode:
    """represents information about a string containing Python code"""

    def __init__(self, code, **exception_kwargs):
        self.code = code
        self.declared_identifiers = set()
        self.undeclared_identifiers = set()
        if isinstance(code, str):
            expr = pyparser.parse(code.lstrip(), 'exec', **exception_kwargs)
        else:
            expr = code
        f = pyparser.FindIdentifiers(self, **exception_kwargs)
        f.visit(expr)