from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
class RuleNode:
    """A node in the parse tree, which also contains the full rhs rule."""

    def __init__(self, rule, children, weight=0):
        self.rule = rule
        self.children = children
        self.weight = weight

    def __repr__(self):
        return 'RuleNode(%s, [%s])' % (repr(self.rule.lhs), ', '.join((str(x) for x in self.children)))