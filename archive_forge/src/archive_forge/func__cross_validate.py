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
def _cross_validate(self, obj: t.Any, value: t.Any) -> G | None:
    if self.name in obj._trait_validators:
        proposal = Bunch({'trait': self, 'value': value, 'owner': obj})
        value = obj._trait_validators[self.name](obj, proposal)
    elif hasattr(obj, '_%s_validate' % self.name):
        meth_name = '_%s_validate' % self.name
        cross_validate = getattr(obj, meth_name)
        deprecated_method(cross_validate, obj.__class__, meth_name, 'use @validate decorator instead.')
        value = cross_validate(value, self)
    return t.cast(G, value)