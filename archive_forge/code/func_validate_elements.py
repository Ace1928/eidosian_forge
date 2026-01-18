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
def validate_elements(self, obj: t.Any, value: dict[t.Any, t.Any]) -> dict[K, V] | None:
    per_key_override = self._per_key_traits or {}
    key_trait = self._key_trait
    value_trait = self._value_trait
    if not (key_trait or value_trait or per_key_override):
        return value
    validated = {}
    for key in value:
        v = value[key]
        if key_trait:
            try:
                key = key_trait._validate(obj, key)
            except TraitError:
                self.element_error(obj, key, key_trait, 'Keys')
        active_value_trait = per_key_override.get(key, value_trait)
        if active_value_trait:
            try:
                v = active_value_trait._validate(obj, v)
            except TraitError:
                self.element_error(obj, v, active_value_trait, 'Values')
        validated[key] = v
    return self.klass(validated)