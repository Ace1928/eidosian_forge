from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
class _CallbackWrapper:
    """An object adapting a on_trait_change callback into an observe callback.

    The comparison operator __eq__ is implemented to enable removal of wrapped
    callbacks.
    """

    def __init__(self, cb: t.Any) -> None:
        self.cb = cb
        offset = -1 if isinstance(self.cb, types.MethodType) else 0
        self.nargs = len(getargspec(cb)[0]) + offset
        if self.nargs > 4:
            raise TraitError('a trait changed callback must have 0-4 arguments.')

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CallbackWrapper):
            return bool(self.cb == other.cb)
        else:
            return bool(self.cb == other)

    def __call__(self, change: Bunch) -> None:
        if self.nargs == 0:
            self.cb()
        elif self.nargs == 1:
            self.cb(change.name)
        elif self.nargs == 2:
            self.cb(change.name, change.new)
        elif self.nargs == 3:
            self.cb(change.name, change.old, change.new)
        elif self.nargs == 4:
            self.cb(change.name, change.old, change.new, change.owner)