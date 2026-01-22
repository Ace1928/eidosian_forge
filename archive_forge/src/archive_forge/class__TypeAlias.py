import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
class _TypeAlias(_TypingBase, _root=True):
    """Internal helper class for defining generic variants of concrete types.

    Note that this is not a type; let's call it a pseudo-type.  It cannot
    be used in instance and subclass checks in parameterized form, i.e.
    ``isinstance(42, Match[str])`` raises ``TypeError`` instead of returning
    ``False``.
    """
    __slots__ = ('name', 'type_var', 'impl_type', 'type_checker')

    def __init__(self, name, type_var, impl_type, type_checker):
        """Initializer.

        Args:
            name: The name, e.g. 'Pattern'.
            type_var: The type parameter, e.g. AnyStr, or the
                specific type, e.g. str.
            impl_type: The implementation type.
            type_checker: Function that takes an impl_type instance.
                and returns a value that should be a type_var instance.
        """
        assert isinstance(name, str), repr(name)
        assert isinstance(impl_type, type), repr(impl_type)
        assert not isinstance(impl_type, TypingMeta), repr(impl_type)
        assert isinstance(type_var, (type, _TypingBase)), repr(type_var)
        self.name = name
        self.type_var = type_var
        self.impl_type = impl_type
        self.type_checker = type_checker

    def __repr__(self):
        return '%s[%s]' % (self.name, _type_repr(self.type_var))

    def __getitem__(self, parameter):
        if not isinstance(self.type_var, TypeVar):
            raise TypeError('%s cannot be further parameterized.' % self)
        if self.type_var.__constraints__ and isinstance(parameter, type):
            if not issubclass(parameter, self.type_var.__constraints__):
                raise TypeError('%s is not a valid substitution for %s.' % (parameter, self.type_var))
        if isinstance(parameter, TypeVar) and parameter is not self.type_var:
            raise TypeError('%s cannot be re-parameterized.' % self)
        return self.__class__(self.name, parameter, self.impl_type, self.type_checker)

    def __eq__(self, other):
        if not isinstance(other, _TypeAlias):
            return NotImplemented
        return self.name == other.name and self.type_var == other.type_var

    def __hash__(self):
        return hash((self.name, self.type_var))

    def __instancecheck__(self, obj):
        if not isinstance(self.type_var, TypeVar):
            raise TypeError('Parameterized type aliases cannot be used with isinstance().')
        return isinstance(obj, self.impl_type)

    def __subclasscheck__(self, cls):
        if not isinstance(self.type_var, TypeVar):
            raise TypeError('Parameterized type aliases cannot be used with issubclass().')
        return issubclass(cls, self.impl_type)