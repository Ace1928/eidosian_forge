from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
def apply_patterns(self, node, patterns):
    for pattern in patterns:
        matcher = pattern()
        if matcher.match(node):
            self.extra_imports.extend(matcher.imports())
            node = matcher.replace()
            self.update = True
    return self.generic_visit(node)