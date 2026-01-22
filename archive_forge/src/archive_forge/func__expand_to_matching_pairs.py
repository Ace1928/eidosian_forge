import ast
import numbers
import sys
import token
from ast import Module
from typing import Callable, List, Union, cast, Optional, Tuple, TYPE_CHECKING
import six
from . import util
from .asttokens import ASTTokens
from .util import AstConstant
from .astroid_compat import astroid_node_classes as nc, BaseContainer as AstroidBaseContainer
def _expand_to_matching_pairs(self, first_token, last_token, node):
    """
    Scan tokens in [first_token, last_token] range that are between node's children, and for any
    unmatched brackets, adjust first/last tokens to include the closing pair.
    """
    to_match_right = []
    to_match_left = []
    for tok in self._code.token_range(first_token, last_token):
        tok_info = tok[:2]
        if to_match_right and tok_info == to_match_right[-1]:
            to_match_right.pop()
        elif tok_info in _matching_pairs_left:
            to_match_right.append(_matching_pairs_left[tok_info])
        elif tok_info in _matching_pairs_right:
            to_match_left.append(_matching_pairs_right[tok_info])
    for match in reversed(to_match_right):
        last = self._code.next_token(last_token)
        while any((util.match_token(last, token.OP, x) for x in (',', ':'))):
            last = self._code.next_token(last)
        if util.match_token(last, *match):
            last_token = last
    for match in to_match_left:
        first = self._code.prev_token(first_token)
        if util.match_token(first, *match):
            first_token = first
    return (first_token, last_token)