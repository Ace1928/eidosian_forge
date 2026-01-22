from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
class Transformer_NonRecursive(Transformer[_Leaf_T, _Return_T]):
    """Same as Transformer but non-recursive.

    Like Transformer, it doesn't change the original tree.

    Useful for huge trees.
    """

    def transform(self, tree: Tree[_Leaf_T]) -> _Return_T:
        rev_postfix = []
        q: List[Branch[_Leaf_T]] = [tree]
        while q:
            t = q.pop()
            rev_postfix.append(t)
            if isinstance(t, Tree):
                q += t.children
        stack: List = []
        for x in reversed(rev_postfix):
            if isinstance(x, Tree):
                size = len(x.children)
                if size:
                    args = stack[-size:]
                    del stack[-size:]
                else:
                    args = []
                res = self._call_userfunc(x, args)
                if res is not Discard:
                    stack.append(res)
            elif self.__visit_tokens__ and isinstance(x, Token):
                res = self._call_userfunc_token(x)
                if res is not Discard:
                    stack.append(res)
            else:
                stack.append(x)
        result, = stack
        return cast(_Return_T, result)