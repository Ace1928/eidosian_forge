import operator
import _ast
from mako import _ast_util
from mako import compat
from mako import exceptions
from mako import util
def _expand_tuples(self, args):
    for arg in args:
        if isinstance(arg, _ast.Tuple):
            yield from arg.elts
        else:
            yield arg