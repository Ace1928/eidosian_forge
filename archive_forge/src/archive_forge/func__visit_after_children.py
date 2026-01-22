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