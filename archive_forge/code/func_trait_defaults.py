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
def trait_defaults(self, *names: str, **metadata: t.Any) -> dict[str, t.Any] | Sentinel:
    """Return a trait's default value or a dictionary of them

        Notes
        -----
        Dynamically generated default values may
        depend on the current state of the object."""
    for n in names:
        if not self.has_trait(n):
            raise TraitError(f"'{n}' is not a trait of '{type(self).__name__}' instances")
    if len(names) == 1 and len(metadata) == 0:
        return t.cast(Sentinel, self._get_trait_default_generator(names[0])(self))
    trait_names = self.trait_names(**metadata)
    trait_names.extend(names)
    defaults = {}
    for n in trait_names:
        defaults[n] = self._get_trait_default_generator(n)(self)
    return defaults