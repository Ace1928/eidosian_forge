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
def _validate_link(*tuples: t.Any) -> None:
    """Validate arguments for traitlet link functions"""
    for tup in tuples:
        if not len(tup) == 2:
            raise TypeError("Each linked traitlet must be specified as (HasTraits, 'trait_name'), not %r" % t)
        obj, trait_name = tup
        if not isinstance(obj, HasTraits):
            raise TypeError('Each object must be HasTraits, not %r' % type(obj))
        if trait_name not in obj.traits():
            raise TypeError(f'{obj!r} has no trait {trait_name!r}')