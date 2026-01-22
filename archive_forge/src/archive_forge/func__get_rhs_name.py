import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _get_rhs_name(node, version):
    type_ = node.type
    if type_ == 'lambdef':
        return 'lambda'
    elif type_ == 'atom':
        comprehension = _get_comprehension_type(node)
        first, second = node.children[:2]
        if comprehension is not None:
            return comprehension
        elif second.type == 'dictorsetmaker':
            if version < (3, 8):
                return 'literal'
            elif second.children[1] == ':' or second.children[0] == '**':
                return 'dict display'
            else:
                return 'set display'
        elif first == '(' and (second == ')' or (len(node.children) == 3 and node.children[1].type == 'testlist_comp')):
            return 'tuple'
        elif first == '(':
            return _get_rhs_name(_remove_parens(node), version=version)
        elif first == '[':
            return 'list'
        elif first == '{' and second == '}':
            return 'dict display'
        elif first == '{' and len(node.children) > 2:
            return 'set display'
    elif type_ == 'keyword':
        if 'yield' in node.value:
            return 'yield expression'
        if version < (3, 8):
            return 'keyword'
        else:
            return str(node.value)
    elif type_ == 'operator' and node.value == '...':
        return 'Ellipsis'
    elif type_ == 'comparison':
        return 'comparison'
    elif type_ in ('string', 'number', 'strings'):
        return 'literal'
    elif type_ == 'yield_expr':
        return 'yield expression'
    elif type_ == 'test':
        return 'conditional expression'
    elif type_ in ('atom_expr', 'power'):
        if node.children[0] == 'await':
            return 'await expression'
        elif node.children[-1].type == 'trailer':
            trailer = node.children[-1]
            if trailer.children[0] == '(':
                return 'function call'
            elif trailer.children[0] == '[':
                return 'subscript'
            elif trailer.children[0] == '.':
                return 'attribute'
    elif 'expr' in type_ and 'star_expr' not in type_ or '_test' in type_ or type_ in ('term', 'factor'):
        return 'operator'
    elif type_ == 'star_expr':
        return 'starred'
    elif type_ == 'testlist_star_expr':
        return 'tuple'
    elif type_ == 'fstring':
        return 'f-string expression'
    return type_