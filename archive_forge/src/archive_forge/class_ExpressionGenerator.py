import operator
import _ast
from mako import _ast_util
from mako import compat
from mako import exceptions
from mako import util
class ExpressionGenerator:

    def __init__(self, astnode):
        self.generator = _ast_util.SourceGenerator(' ' * 4)
        self.generator.visit(astnode)

    def value(self):
        return ''.join(self.generator.result)