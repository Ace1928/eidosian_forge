import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
class TypeAliasType:
    """Create named, parameterized type aliases.

        This provides a backport of the new `type` statement in Python 3.12:

            type ListOrSet[T] = list[T] | set[T]

        is equivalent to:

            T = TypeVar("T")
            ListOrSet = TypeAliasType("ListOrSet", list[T] | set[T], type_params=(T,))

        The name ListOrSet can then be used as an alias for the type it refers to.

        The type_params argument should contain all the type parameters used
        in the value of the type alias. If the alias is not generic, this
        argument is omitted.

        Static type checkers should only support type aliases declared using
        TypeAliasType that follow these rules:

        - The first argument (the name) must be a string literal.
        - The TypeAliasType instance must be immediately assigned to a variable
          of the same name. (For example, 'X = TypeAliasType("Y", int)' is invalid,
          as is 'X, Y = TypeAliasType("X", int), TypeAliasType("Y", int)').

        """

    def __init__(self, name: str, value, *, type_params=()):
        if not isinstance(name, str):
            raise TypeError('TypeAliasType name must be a string')
        self.__value__ = value
        self.__type_params__ = type_params
        parameters = []
        for type_param in type_params:
            if isinstance(type_param, TypeVarTuple):
                parameters.extend(type_param)
            else:
                parameters.append(type_param)
        self.__parameters__ = tuple(parameters)
        def_mod = _caller()
        if def_mod != 'typing_extensions':
            self.__module__ = def_mod
        self.__name__ = name

    def __setattr__(self, name: str, value: object, /) -> None:
        if hasattr(self, '__name__'):
            self._raise_attribute_error(name)
        super().__setattr__(name, value)

    def __delattr__(self, name: str, /) -> Never:
        self._raise_attribute_error(name)

    def _raise_attribute_error(self, name: str) -> Never:
        if name == '__name__':
            raise AttributeError('readonly attribute')
        elif name in {'__value__', '__type_params__', '__parameters__', '__module__'}:
            raise AttributeError(f"attribute '{name}' of 'typing.TypeAliasType' objects is not writable")
        else:
            raise AttributeError(f"'typing.TypeAliasType' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return self.__name__

    def __getitem__(self, parameters):
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        parameters = [typing._type_check(item, f'Subscripting {self.__name__} requires a type.') for item in parameters]
        return typing._GenericAlias(self, tuple(parameters))

    def __reduce__(self):
        return self.__name__

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError("type 'typing_extensions.TypeAliasType' is not an acceptable base type")

    def __call__(self):
        raise TypeError('Type alias is not callable')
    if sys.version_info >= (3, 10):

        def __or__(self, right):
            if not _is_unionable(right):
                return NotImplemented
            return typing.Union[self, right]

        def __ror__(self, left):
            if not _is_unionable(left):
                return NotImplemented
            return typing.Union[left, self]