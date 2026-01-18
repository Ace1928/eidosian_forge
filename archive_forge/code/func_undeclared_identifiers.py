import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
def undeclared_identifiers(self):
    return self.code.undeclared_identifiers.difference(self.code.declared_identifiers)