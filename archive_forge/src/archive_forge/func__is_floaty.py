import re
from lib2to3.fixer_util import Leaf, Node, Comma
from lib2to3 import fixer_base
from libfuturize.fixer_util import (token, future_import, touch_import_top,
def _is_floaty(expr):
    if isinstance(expr, list):
        expr = expr[0]
    if isinstance(expr, Leaf):
        return const_re.match(expr.value)
    elif isinstance(expr, Node):
        if isinstance(expr.children[0], Leaf):
            return expr.children[0].value == u'float'
    return False