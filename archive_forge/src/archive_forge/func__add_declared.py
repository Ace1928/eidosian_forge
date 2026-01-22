import operator
import _ast
from mako import _ast_util
from mako import compat
from mako import exceptions
from mako import util
def _add_declared(self, name):
    if not self.in_function:
        self.listener.declared_identifiers.add(name)
    else:
        self.local_ident_stack.add(name)