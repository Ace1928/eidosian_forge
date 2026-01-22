import contextvars
from functools import singledispatch
import os
from typing import Any
from typing import Optional
import typing
import warnings
from rpy2.rinterface_lib import _rinterface_capi
import rpy2.rinterface_lib.sexp
import rpy2.rinterface_lib.conversion
import rpy2.rinterface
class NameClassMap(object):
    """Map a name (R class name) to a Python class.

    R class names, as returned for example by the R function
    `class()`, are arrays of strings representing the
    class lineage. This class helps mapping the class of an R
    object (a sequence of names) to a Python class.

    For example, R data frames are of class "data.frame", but are
    R lists (VECSEXP) at the C level. The NameClassMap for that
    such R VECSEXP objects would be:

    NameClassMap(robjects.vectors.ListVector,
                 {'data.frame': robjects.vectors.DataFrame})

    This means that the default class on the Python side will be
    `ListVector`, but if the R object is a "data.frame" it will
    be a `DataFrame`.
    """
    _default: typing.Union[typing.Any, typing.Callable[[typing.Any], typing.Any]]
    _map: typing.Dict[str, typing.Union[typing.Any, typing.Callable[[typing.Any], typing.Any]]]
    default = property(lambda self: self._default)

    def __init__(self, defaultcls: typing.Union[typing.Type, typing.Callable[[typing.Any], typing.Any]]=object, namemap: typing.Optional[dict]=None):
        if namemap is None:
            namemap = {}
        self._default = defaultcls
        self._map = namemap.copy()

    def __contains__(self, key: str) -> bool:
        return key in self._map

    def __delitem__(self, key: str) -> None:
        del self._map[key]

    def __getitem__(self, key: str) -> typing.Union[typing.Type, typing.Callable[[typing.Any], typing.Any]]:
        return self._map[key]

    def __setitem__(self, key: str, value: typing.Union[typing.Type[typing.Any], typing.Callable[[typing.Any], typing.Any]]):
        self._map[key] = value

    def copy(self) -> 'NameClassMap':
        return NameClassMap(defaultcls=self._default, namemap=self._map.copy())

    def update(self, mapping: typing.Dict[str, typing.Union[typing.Any, typing.Callable[[typing.Any], typing.Any]]], default: typing.Optional[typing.Type]=None):
        self._map.update(mapping)
        if default:
            self._default = default

    def find_key(self, keys: typing.Iterable[str]) -> typing.Optional[str]:
        """
        Find the first mapping key in a sequence of names (keys).

        Args:
          keys (iterable): The keys are the R classes (the last being the
            most distant ancestor class)
        Returns:
           None if no mapping key.
        """
        for k in keys:
            if k in self._map:
                return k
        return None

    def find(self, keys: typing.Iterable[str]) -> typing.Union[typing.Type, typing.Callable[[typing.Any], typing.Any]]:
        """Find the first mapping in a sequence of names (keys).

        Returns the default class (specified when creating the
        instance if no mapping key."""
        k = self.find_key(keys)
        if k:
            cls = self._map[k]
        else:
            cls = self._default
        return cls