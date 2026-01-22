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
def add_traits(self, **traits: t.Any) -> None:
    """Dynamically add trait attributes to the HasTraits instance."""
    cls = self.__class__
    attrs = {'__module__': cls.__module__}
    if hasattr(cls, '__qualname__'):
        attrs['__qualname__'] = cls.__qualname__
    attrs.update(traits)
    self.__class__ = type(cls.__name__, (cls,), attrs)
    for trait in traits.values():
        trait.instance_init(self)