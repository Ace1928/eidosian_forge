from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class Pattern(object):

    def match(self, node):
        self.check = Check(node, dict())
        return self.check.visit(self.pattern)

    def replace(self):
        return PlaceholderReplace(self.check.placeholders).visit(self.sub())

    def imports(self):
        return deepcopy(getattr(self, 'extra_imports', []))