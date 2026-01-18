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
def visit_const(self, node, first_token, last_token):
    assert isinstance(node, AstConstant) or isinstance(node, nc.Const)
    if isinstance(node.value, numbers.Number):
        return self.handle_num(node, node.value, first_token, last_token)
    elif isinstance(node.value, (six.text_type, six.binary_type)):
        return self.visit_str(node, first_token, last_token)
    return (first_token, last_token)