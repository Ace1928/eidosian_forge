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
class DefaultHandler(EventHandler):

    def __init__(self, name: str) -> None:
        self.trait_name = name

    def class_init(self, cls: type[HasTraits], name: str | None) -> None:
        super().class_init(cls, name)
        cls._trait_default_generators[self.trait_name] = self