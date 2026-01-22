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
class _GenericAlias(_BaseGenericAlias, _root=True):

    def __init__(self, origin, args, *, inst=True, name=None, _paramspec_tvars=False):
        super().__init__(origin, inst=inst, name=name)
        if not isinstance(args, tuple):
            args = (args,)
        self.__args__ = tuple((... if a is _TypingEllipsis else a for a in args))
        self.__parameters__ = _collect_parameters(args)
        self._paramspec_tvars = _paramspec_tvars
        if not name:
            self.__module__ = origin.__module__

    def __eq__(self, other):
        if not isinstance(other, _GenericAlias):
            return NotImplemented
        return self.__origin__ == other.__origin__ and self.__args__ == other.__args__

    def __hash__(self):
        return hash((self.__origin__, self.__args__))

    def __or__(self, right):
        return Union[self, right]

    def __ror__(self, left):
        return Union[left, self]

    @_tp_cache
    def __getitem__(self, args):
        if self.__origin__ in (Generic, Protocol):
            raise TypeError(f'Cannot subscript already-subscripted {self}')
        if not self.__parameters__:
            raise TypeError(f'{self} is not a generic class')
        if not isinstance(args, tuple):
            args = (args,)
        args = tuple((_type_convert(p) for p in args))
        args = _unpack_args(args)
        new_args = self._determine_new_args(args)
        r = self.copy_with(new_args)
        return r

    def _determine_new_args(self, args):
        params = self.__parameters__
        for param in params:
            prepare = getattr(param, '__typing_prepare_subst__', None)
            if prepare is not None:
                args = prepare(self, args)
        alen = len(args)
        plen = len(params)
        if alen != plen:
            raise TypeError(f'Too {('many' if alen > plen else 'few')} arguments for {self}; actual {alen}, expected {plen}')
        new_arg_by_param = dict(zip(params, args))
        return tuple(self._make_substitution(self.__args__, new_arg_by_param))

    def _make_substitution(self, args, new_arg_by_param):
        """Create a list of new type arguments."""
        new_args = []
        for old_arg in args:
            if isinstance(old_arg, type):
                new_args.append(old_arg)
                continue
            substfunc = getattr(old_arg, '__typing_subst__', None)
            if substfunc:
                new_arg = substfunc(new_arg_by_param[old_arg])
            else:
                subparams = getattr(old_arg, '__parameters__', ())
                if not subparams:
                    new_arg = old_arg
                else:
                    subargs = []
                    for x in subparams:
                        if isinstance(x, TypeVarTuple):
                            subargs.extend(new_arg_by_param[x])
                        else:
                            subargs.append(new_arg_by_param[x])
                    new_arg = old_arg[tuple(subargs)]
            if self.__origin__ == collections.abc.Callable and isinstance(new_arg, tuple):
                new_args.extend(new_arg)
            elif _is_unpacked_typevartuple(old_arg):
                new_args.extend(new_arg)
            elif isinstance(old_arg, tuple):
                new_args.append(tuple(self._make_substitution(old_arg, new_arg_by_param)))
            else:
                new_args.append(new_arg)
        return new_args

    def copy_with(self, args):
        return self.__class__(self.__origin__, args, name=self._name, inst=self._inst, _paramspec_tvars=self._paramspec_tvars)

    def __repr__(self):
        if self._name:
            name = 'typing.' + self._name
        else:
            name = _type_repr(self.__origin__)
        if self.__args__:
            args = ', '.join([_type_repr(a) for a in self.__args__])
        else:
            args = '()'
        return f'{name}[{args}]'

    def __reduce__(self):
        if self._name:
            origin = globals()[self._name]
        else:
            origin = self.__origin__
        args = tuple(self.__args__)
        if len(args) == 1 and (not isinstance(args[0], tuple)):
            args, = args
        return (operator.getitem, (origin, args))

    def __mro_entries__(self, bases):
        if isinstance(self.__origin__, _SpecialForm):
            raise TypeError(f'Cannot subclass {self!r}')
        if self._name:
            return super().__mro_entries__(bases)
        if self.__origin__ is Generic:
            if Protocol in bases:
                return ()
            i = bases.index(self)
            for b in bases[i + 1:]:
                if isinstance(b, _BaseGenericAlias) and b is not self:
                    return ()
        return (self.__origin__,)

    def __iter__(self):
        yield Unpack[self]