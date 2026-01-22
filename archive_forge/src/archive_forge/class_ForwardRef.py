from abc import abstractmethod, ABCMeta
import collections
from collections import defaultdict
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import warnings
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, GenericAlias
class ForwardRef(_Final, _root=True):
    """Internal wrapper to hold a forward reference."""
    __slots__ = ('__forward_arg__', '__forward_code__', '__forward_evaluated__', '__forward_value__', '__forward_is_argument__', '__forward_is_class__', '__forward_module__')

    def __init__(self, arg, is_argument=True, module=None, *, is_class=False):
        if not isinstance(arg, str):
            raise TypeError(f'Forward reference must be a string -- got {arg!r}')
        if arg[0] == '*':
            arg_to_compile = f'({arg},)[0]'
        else:
            arg_to_compile = arg
        try:
            code = compile(arg_to_compile, '<string>', 'eval')
        except SyntaxError:
            raise SyntaxError(f'Forward reference must be an expression -- got {arg!r}')
        self.__forward_arg__ = arg
        self.__forward_code__ = code
        self.__forward_evaluated__ = False
        self.__forward_value__ = None
        self.__forward_is_argument__ = is_argument
        self.__forward_is_class__ = is_class
        self.__forward_module__ = module

    def _evaluate(self, globalns, localns, recursive_guard):
        if self.__forward_arg__ in recursive_guard:
            return self
        if not self.__forward_evaluated__ or localns is not globalns:
            if globalns is None and localns is None:
                globalns = localns = {}
            elif globalns is None:
                globalns = localns
            elif localns is None:
                localns = globalns
            if self.__forward_module__ is not None:
                globalns = getattr(sys.modules.get(self.__forward_module__, None), '__dict__', globalns)
            type_ = _type_check(eval(self.__forward_code__, globalns, localns), 'Forward references must evaluate to types.', is_argument=self.__forward_is_argument__, allow_special_forms=self.__forward_is_class__)
            self.__forward_value__ = _eval_type(type_, globalns, localns, recursive_guard | {self.__forward_arg__})
            self.__forward_evaluated__ = True
        return self.__forward_value__

    def __eq__(self, other):
        if not isinstance(other, ForwardRef):
            return NotImplemented
        if self.__forward_evaluated__ and other.__forward_evaluated__:
            return self.__forward_arg__ == other.__forward_arg__ and self.__forward_value__ == other.__forward_value__
        return self.__forward_arg__ == other.__forward_arg__ and self.__forward_module__ == other.__forward_module__

    def __hash__(self):
        return hash((self.__forward_arg__, self.__forward_module__))

    def __or__(self, other):
        return Union[self, other]

    def __ror__(self, other):
        return Union[other, self]

    def __repr__(self):
        if self.__forward_module__ is None:
            module_repr = ''
        else:
            module_repr = f', module={self.__forward_module__!r}'
        return f'ForwardRef({self.__forward_arg__!r}{module_repr})'