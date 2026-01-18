from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
@staticmethod
def visit_AST_any(_):
    """ Every node match with it. """
    return True