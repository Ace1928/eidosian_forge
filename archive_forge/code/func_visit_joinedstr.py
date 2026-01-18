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