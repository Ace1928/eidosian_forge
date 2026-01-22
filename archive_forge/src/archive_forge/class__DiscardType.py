from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
class _DiscardType:
    """When the Discard value is returned from a transformer callback,
    that node is discarded and won't appear in the parent.

    Note:
        This feature is disabled when the transformer is provided to Lark
        using the ``transformer`` keyword (aka Tree-less LALR mode).

    Example:
        ::

            class T(Transformer):
                def ignore_tree(self, children):
                    return Discard

                def IGNORE_TOKEN(self, token):
                    return Discard
    """

    def __repr__(self):
        return 'lark.visitors.Discard'