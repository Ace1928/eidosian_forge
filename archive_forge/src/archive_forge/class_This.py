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
class This(ClassBasedTraitType[t.Optional[T], t.Optional[T]]):
    """A trait for instances of the class containing this trait.

    Because how how and when class bodies are executed, the ``This``
    trait can only have a default value of None.  This, and because we
    always validate default values, ``allow_none`` is *always* true.
    """
    info_text = 'an instance of the same type as the receiver or None'

    def __init__(self, **kwargs: t.Any) -> None:
        super().__init__(None, **kwargs)

    def validate(self, obj: t.Any, value: t.Any) -> HasTraits | None:
        assert self.this_class is not None
        if isinstance(value, self.this_class) or value is None:
            return value
        else:
            self.error(obj, value)