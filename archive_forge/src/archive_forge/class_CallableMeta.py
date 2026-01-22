import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
class CallableMeta(GenericMeta):
    """Metaclass for Callable (internal)."""

    def __repr__(self):
        if self.__origin__ is None:
            return super().__repr__()
        return self._tree_repr(self._subs_tree())

    def _tree_repr(self, tree):
        if self._gorg is not Callable:
            return super()._tree_repr(tree)
        arg_list = []
        for arg in tree[1:]:
            if not isinstance(arg, tuple):
                arg_list.append(_type_repr(arg))
            else:
                arg_list.append(arg[0]._tree_repr(arg))
        if arg_list[0] == '...':
            return repr(tree[0]) + '[..., %s]' % arg_list[1]
        return repr(tree[0]) + '[[%s], %s]' % (', '.join(arg_list[:-1]), arg_list[-1])

    def __getitem__(self, parameters):
        """A thin wrapper around __getitem_inner__ to provide the latter
        with hashable arguments to improve speed.
        """
        if self.__origin__ is not None or self._gorg is not Callable:
            return super().__getitem__(parameters)
        if not isinstance(parameters, tuple) or len(parameters) != 2:
            raise TypeError('Callable must be used as Callable[[arg, ...], result].')
        args, result = parameters
        if args is Ellipsis:
            parameters = (Ellipsis, result)
        else:
            if not isinstance(args, list):
                raise TypeError('Callable[args, result]: args must be a list. Got %.100r.' % (args,))
            parameters = (tuple(args), result)
        return self.__getitem_inner__(parameters)

    @_tp_cache
    def __getitem_inner__(self, parameters):
        args, result = parameters
        msg = 'Callable[args, result]: result must be a type.'
        result = _type_check(result, msg)
        if args is Ellipsis:
            return super().__getitem__((_TypingEllipsis, result))
        msg = 'Callable[[arg, ...], result]: each arg must be a type.'
        args = tuple((_type_check(arg, msg) for arg in args))
        parameters = args + (result,)
        return super().__getitem__(parameters)