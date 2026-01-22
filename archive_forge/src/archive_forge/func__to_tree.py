from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def _to_tree(self, rule_node):
    """Converts a RuleNode parse tree to a lark Tree."""
    orig_rule = self.orig_rules[rule_node.rule.alias]
    children = []
    for child in rule_node.children:
        if isinstance(child, RuleNode):
            children.append(self._to_tree(child))
        else:
            assert isinstance(child.name, Token)
            children.append(child.name)
    t = Tree(orig_rule.origin, children)
    t.rule = orig_rule
    return t