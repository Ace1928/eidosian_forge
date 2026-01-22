import abc
import collections
import collections.abc
import operator
import sys
import typing
class _MaybeRequired(typing._FinalTypingBase, _root=True):
    __slots__ = ('__type__',)

    def __init__(self, tp=None, **kwds):
        self.__type__ = tp

    def __getitem__(self, item):
        cls = type(self)
        if self.__type__ is None:
            return cls(typing._type_check(item, '{} accepts only single type.'.format(cls.__name__[1:])), _root=True)
        raise TypeError('{} cannot be further subscripted'.format(cls.__name__[1:]))

    def _eval_type(self, globalns, localns):
        new_tp = typing._eval_type(self.__type__, globalns, localns)
        if new_tp == self.__type__:
            return self
        return type(self)(new_tp, _root=True)

    def __repr__(self):
        r = super().__repr__()
        if self.__type__ is not None:
            r += '[{}]'.format(typing._type_repr(self.__type__))
        return r

    def __hash__(self):
        return hash((type(self).__name__, self.__type__))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.__type__ is not None:
            return self.__type__ == other.__type__
        return self is other