from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class PatternTransform(Transformation):
    """
    Replace all known pattern by pythran function call.

    Based on BaseMatcher to search correct pattern.
    """

    def __init__(self):
        """ Initialize the Basematcher to search for placeholders. """
        super(PatternTransform, self).__init__()

    def visit_Module(self, node):
        self.extra_imports = []
        self.generic_visit(node)
        node.body = self.extra_imports + node.body
        return node

    def apply_patterns(self, node, patterns):
        for pattern in patterns:
            matcher = pattern()
            if matcher.match(node):
                self.extra_imports.extend(matcher.imports())
                node = matcher.replace()
                self.update = True
        return self.generic_visit(node)
    CallPatterns = ()

    def visit_Call(self, node):
        return self.apply_patterns(node, PatternTransform.CallPatterns)
    BinOpPatterns = ()

    def visit_BinOp(self, node):
        return self.apply_patterns(node, PatternTransform.BinOpPatterns)