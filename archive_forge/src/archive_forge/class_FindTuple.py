import operator
import _ast
from mako import _ast_util
from mako import compat
from mako import exceptions
from mako import util
class FindTuple(_ast_util.NodeVisitor):

    def __init__(self, listener, code_factory, **exception_kwargs):
        self.listener = listener
        self.exception_kwargs = exception_kwargs
        self.code_factory = code_factory

    def visit_Tuple(self, node):
        for n in node.elts:
            p = self.code_factory(n, **self.exception_kwargs)
            self.listener.codeargs.append(p)
            self.listener.args.append(ExpressionGenerator(n).value())
            ldi = self.listener.declared_identifiers
            self.listener.declared_identifiers = ldi.union(p.declared_identifiers)
            lui = self.listener.undeclared_identifiers
            self.listener.undeclared_identifiers = lui.union(p.undeclared_identifiers)