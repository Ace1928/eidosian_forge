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
class MarkTokens(object):
    """
  Helper that visits all nodes in the AST tree and assigns .first_token and .last_token attributes
  to each of them. This is the heart of the token-marking logic.
  """

    def __init__(self, code):
        self._code = code
        self._methods = util.NodeMethods()
        self._iter_children = None

    def visit_tree(self, node):
        self._iter_children = util.iter_children_func(node)
        util.visit_tree(node, self._visit_before_children, self._visit_after_children)

    def _visit_before_children(self, node, parent_token):
        col = getattr(node, 'col_offset', None)
        token = self._code.get_token_from_utf8(node.lineno, col) if col is not None else None
        if not token and util.is_module(node):
            token = self._code.get_token(1, 0)
        return (token or parent_token, token)

    def _visit_after_children(self, node, parent_token, token):
        first = token
        last = None
        for child in cast(Callable, self._iter_children)(node):
            if util.is_empty_astroid_slice(child):
                continue
            if not first or child.first_token.index < first.index:
                first = child.first_token
            if not last or child.last_token.index > last.index:
                last = child.last_token
        first = first or parent_token
        last = last or first
        if util.is_stmt(node):
            last = self._find_last_in_stmt(cast(util.Token, last))
        first, last = self._expand_to_matching_pairs(cast(util.Token, first), cast(util.Token, last), node)
        nfirst, nlast = self._methods.get(self, node.__class__)(node, first, last)
        if (nfirst, nlast) != (first, last):
            nfirst, nlast = self._expand_to_matching_pairs(nfirst, nlast, node)
        node.first_token = nfirst
        node.last_token = nlast

    def _find_last_in_stmt(self, start_token):
        t = start_token
        while not util.match_token(t, token.NEWLINE) and (not util.match_token(t, token.OP, ';')) and (not token.ISEOF(t.type)):
            t = self._code.next_token(t, include_extra=True)
        return self._code.prev_token(t)

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

    def visit_default(self, node, first_token, last_token):
        return (first_token, last_token)

    def handle_comp(self, open_brace, node, first_token, last_token):
        before = self._code.prev_token(first_token)
        util.expect_token(before, token.OP, open_brace)
        return (before, last_token)
    if sys.version_info < (3, 8):

        def visit_listcomp(self, node, first_token, last_token):
            return self.handle_comp('[', node, first_token, last_token)
    if six.PY2:

        def visit_setcomp(self, node, first_token, last_token):
            return self.handle_comp('{', node, first_token, last_token)

        def visit_dictcomp(self, node, first_token, last_token):
            return self.handle_comp('{', node, first_token, last_token)

    def visit_comprehension(self, node, first_token, last_token):
        first = self._code.find_token(first_token, token.NAME, 'for', reverse=True)
        return (first, last_token)

    def visit_if(self, node, first_token, last_token):
        while first_token.string not in ('if', 'elif'):
            first_token = self._code.prev_token(first_token)
        return (first_token, last_token)

    def handle_attr(self, node, first_token, last_token):
        dot = self._code.find_token(last_token, token.OP, '.')
        name = self._code.next_token(dot)
        util.expect_token(name, token.NAME)
        return (first_token, name)
    visit_attribute = handle_attr
    visit_assignattr = handle_attr
    visit_delattr = handle_attr

    def handle_def(self, node, first_token, last_token):
        if not node.body and (getattr(node, 'doc_node', None) or getattr(node, 'doc', None)):
            last_token = self._code.find_token(last_token, token.STRING)
        if first_token.index > 0:
            prev = self._code.prev_token(first_token)
            if util.match_token(prev, token.OP, '@'):
                first_token = prev
        return (first_token, last_token)
    visit_classdef = handle_def
    visit_functiondef = handle_def

    def handle_following_brackets(self, node, last_token, opening_bracket):
        first_child = next(cast(Callable, self._iter_children)(node))
        call_start = self._code.find_token(first_child.last_token, token.OP, opening_bracket)
        if call_start.index > last_token.index:
            last_token = call_start
        return last_token

    def visit_call(self, node, first_token, last_token):
        last_token = self.handle_following_brackets(node, last_token, '(')
        if util.match_token(first_token, token.OP, '@'):
            first_token = self._code.next_token(first_token)
        return (first_token, last_token)

    def visit_matchclass(self, node, first_token, last_token):
        last_token = self.handle_following_brackets(node, last_token, '(')
        return (first_token, last_token)

    def visit_subscript(self, node, first_token, last_token):
        last_token = self.handle_following_brackets(node, last_token, '[')
        return (first_token, last_token)

    def visit_slice(self, node, first_token, last_token):
        while True:
            prev = self._code.prev_token(first_token)
            if prev.string != ':':
                break
            first_token = prev
        while True:
            next_ = self._code.next_token(last_token)
            if next_.string != ':':
                break
            last_token = next_
        return (first_token, last_token)

    def handle_bare_tuple(self, node, first_token, last_token):
        maybe_comma = self._code.next_token(last_token)
        if util.match_token(maybe_comma, token.OP, ','):
            last_token = maybe_comma
        return (first_token, last_token)
    if sys.version_info >= (3, 8):

        def handle_tuple_nonempty(self, node, first_token, last_token):
            assert isinstance(node, ast.Tuple) or isinstance(node, AstroidBaseContainer)
            child = node.elts[0]
            if TYPE_CHECKING:
                child = cast(AstNode, child)
            child_first, child_last = self._gobble_parens(child.first_token, child.last_token, True)
            if first_token == child_first:
                return self.handle_bare_tuple(node, first_token, last_token)
            return (first_token, last_token)
    else:

        def handle_tuple_nonempty(self, node, first_token, last_token):
            first_token, last_token = self.handle_bare_tuple(node, first_token, last_token)
            return self._gobble_parens(first_token, last_token, False)

    def visit_tuple(self, node, first_token, last_token):
        assert isinstance(node, ast.Tuple) or isinstance(node, AstroidBaseContainer)
        if not node.elts:
            return (first_token, last_token)
        return self.handle_tuple_nonempty(node, first_token, last_token)

    def _gobble_parens(self, first_token, last_token, include_all=False):
        while first_token.index > 0:
            prev = self._code.prev_token(first_token)
            next = self._code.next_token(last_token)
            if util.match_token(prev, token.OP, '(') and util.match_token(next, token.OP, ')'):
                first_token, last_token = (prev, next)
                if include_all:
                    continue
            break
        return (first_token, last_token)

    def visit_str(self, node, first_token, last_token):
        return self.handle_str(first_token, last_token)

    def visit_joinedstr(self, node, first_token, last_token):
        if sys.version_info < (3, 12):
            return self.handle_str(first_token, last_token)
        last = first_token
        while True:
            if util.match_token(last, getattr(token, 'FSTRING_START')):
                count = 1
                while count > 0:
                    last = self._code.next_token(last)
                    if util.match_token(last, getattr(token, 'FSTRING_START')):
                        count += 1
                    elif util.match_token(last, getattr(token, 'FSTRING_END')):
                        count -= 1
                last_token = last
                last = self._code.next_token(last_token)
            elif util.match_token(last, token.STRING):
                last_token = last
                last = self._code.next_token(last_token)
            else:
                break
        return (first_token, last_token)

    def visit_bytes(self, node, first_token, last_token):
        return self.handle_str(first_token, last_token)

    def handle_str(self, first_token, last_token):
        last = self._code.next_token(last_token)
        while util.match_token(last, token.STRING):
            last_token = last
            last = self._code.next_token(last_token)
        return (first_token, last_token)

    def handle_num(self, node, value, first_token, last_token):
        while util.match_token(last_token, token.OP):
            last_token = self._code.next_token(last_token)
        if isinstance(value, complex):
            value = value.imag
        if value < 0 and first_token.type == token.NUMBER:
            first_token = self._code.prev_token(first_token)
        return (first_token, last_token)

    def visit_num(self, node, first_token, last_token):
        return self.handle_num(node, cast(ast.Num, node).n, first_token, last_token)

    def visit_const(self, node, first_token, last_token):
        assert isinstance(node, AstConstant) or isinstance(node, nc.Const)
        if isinstance(node.value, numbers.Number):
            return self.handle_num(node, node.value, first_token, last_token)
        elif isinstance(node.value, (six.text_type, six.binary_type)):
            return self.visit_str(node, first_token, last_token)
        return (first_token, last_token)
    visit_constant = visit_const

    def visit_keyword(self, node, first_token, last_token):
        assert isinstance(node, ast.keyword) or isinstance(node, nc.Keyword)
        if node.arg is not None and getattr(node, 'lineno', None) is None:
            equals = self._code.find_token(first_token, token.OP, '=', reverse=True)
            name = self._code.prev_token(equals)
            util.expect_token(name, token.NAME, node.arg)
            first_token = name
        return (first_token, last_token)

    def visit_starred(self, node, first_token, last_token):
        if not util.match_token(first_token, token.OP, '*'):
            star = self._code.prev_token(first_token)
            if util.match_token(star, token.OP, '*'):
                first_token = star
        return (first_token, last_token)

    def visit_assignname(self, node, first_token, last_token):
        if util.match_token(first_token, token.NAME, 'except'):
            colon = self._code.find_token(last_token, token.OP, ':')
            first_token = last_token = self._code.prev_token(colon)
        return (first_token, last_token)
    if six.PY2:

        def visit_with(self, node, first_token, last_token):
            first = self._code.find_token(first_token, token.NAME, 'with', reverse=True)
            return (first, last_token)

    def handle_async(self, node, first_token, last_token):
        if not first_token.string == 'async':
            first_token = self._code.prev_token(first_token)
        return (first_token, last_token)
    visit_asyncfor = handle_async
    visit_asyncwith = handle_async

    def visit_asyncfunctiondef(self, node, first_token, last_token):
        if util.match_token(first_token, token.NAME, 'def'):
            first_token = self._code.prev_token(first_token)
        return self.visit_functiondef(node, first_token, last_token)