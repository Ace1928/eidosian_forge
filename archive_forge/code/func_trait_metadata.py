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
def trait_metadata(self, traitname: str, key: str, default: t.Any=None) -> t.Any:
    """Get metadata values for trait by key."""
    try:
        trait = getattr(self.__class__, traitname)
    except AttributeError as e:
        raise TraitError(f'Class {self.__class__.__name__} does not have a trait named {traitname}') from e
    metadata_name = '_' + traitname + '_metadata'
    if hasattr(self, metadata_name) and key in getattr(self, metadata_name):
        return getattr(self, metadata_name).get(key, default)
    else:
        return trait.metadata.get(key, default)