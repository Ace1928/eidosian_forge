import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
class _CountingAttr:
    """
    Intermediate representation of attributes that uses a counter to preserve
    the order in which the attributes have been defined.

    *Internal* data structure of the attrs library.  Running into is most
    likely the result of a bug like a forgotten `@attr.s` decorator.
    """
    __slots__ = ('counter', '_default', 'repr', 'eq', 'eq_key', 'order', 'order_key', 'hash', 'init', 'metadata', '_validator', 'converter', 'type', 'kw_only', 'on_setattr', 'alias')
    __attrs_attrs__ = (*tuple((Attribute(name=name, alias=_default_init_alias_for(name), default=NOTHING, validator=None, repr=True, cmp=None, hash=True, init=True, kw_only=False, eq=True, eq_key=None, order=False, order_key=None, inherited=False, on_setattr=None) for name in ('counter', '_default', 'repr', 'eq', 'order', 'hash', 'init', 'on_setattr', 'alias'))), Attribute(name='metadata', alias='metadata', default=None, validator=None, repr=True, cmp=None, hash=False, init=True, kw_only=False, eq=True, eq_key=None, order=False, order_key=None, inherited=False, on_setattr=None))
    cls_counter = 0

    def __init__(self, default, validator, repr, cmp, hash, init, converter, metadata, type, kw_only, eq, eq_key, order, order_key, on_setattr, alias):
        _CountingAttr.cls_counter += 1
        self.counter = _CountingAttr.cls_counter
        self._default = default
        self._validator = validator
        self.converter = converter
        self.repr = repr
        self.eq = eq
        self.eq_key = eq_key
        self.order = order
        self.order_key = order_key
        self.hash = hash
        self.init = init
        self.metadata = metadata
        self.type = type
        self.kw_only = kw_only
        self.on_setattr = on_setattr
        self.alias = alias

    def validator(self, meth):
        """
        Decorator that adds *meth* to the list of validators.

        Returns *meth* unchanged.

        .. versionadded:: 17.1.0
        """
        if self._validator is None:
            self._validator = meth
        else:
            self._validator = and_(self._validator, meth)
        return meth

    def default(self, meth):
        """
        Decorator that allows to set the default for an attribute.

        Returns *meth* unchanged.

        :raises DefaultAlreadySetError: If default has been set before.

        .. versionadded:: 17.1.0
        """
        if self._default is not NOTHING:
            raise DefaultAlreadySetError()
        self._default = Factory(meth, takes_self=True)
        return meth