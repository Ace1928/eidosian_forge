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
def _find_last_in_stmt(self, start_token):
    t = start_token
    while not util.match_token(t, token.NEWLINE) and (not util.match_token(t, token.OP, ';')) and (not token.ISEOF(t.type)):
        t = self._code.next_token(t, include_extra=True)
    return self._code.prev_token(t)