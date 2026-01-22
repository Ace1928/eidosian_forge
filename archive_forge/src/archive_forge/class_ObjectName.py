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
class ObjectName(TraitType[str, str]):
    """A string holding a valid object name in this version of Python.

    This does not check that the name exists in any scope."""
    info_text = 'a valid object identifier in Python'
    coerce_str = staticmethod(lambda _, s: s)

    def validate(self, obj: t.Any, value: t.Any) -> str:
        value = self.coerce_str(obj, value)
        if isinstance(value, str) and isidentifier(value):
            return value
        self.error(obj, value)

    def from_string(self, s: str) -> str | None:
        if self.allow_none and s == 'None':
            return None
        return s