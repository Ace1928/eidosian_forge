from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
class Placeholder(AST):
    """ Class to save information from ast while check for pattern. """

    def __init__(self, identifier, type=None, constraint=None):
        """ Placeholder are identified using an identifier. """
        self.id = identifier
        self.type = type
        self.constraint = constraint
        super(Placeholder, self).__init__()