from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
def visit_Placeholder(self, pattern):
    """
        Save matching node or compare it with the existing one.

        FIXME : What if the new placeholder is a better choice?
        """
    if pattern.id in self.placeholders and (not Check(self.node, self.placeholders).visit(self.placeholders[pattern.id])):
        return False
    elif pattern.type is not None and (not isinstance(self.node, pattern.type)):
        return False
    elif pattern.constraint is not None and (not pattern.constraint(self.node)):
        return False
    else:
        self.placeholders[pattern.id] = self.node
        return True