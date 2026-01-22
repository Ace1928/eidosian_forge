from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def _to_rule(self, lark_rule):
    """Converts a lark rule, (lhs, rhs, callback, options), to a Rule."""
    assert isinstance(lark_rule.origin, NT)
    assert all((isinstance(x, Symbol) for x in lark_rule.expansion))
    return Rule(lark_rule.origin, lark_rule.expansion, weight=lark_rule.options.priority if lark_rule.options.priority else 0, alias=lark_rule)