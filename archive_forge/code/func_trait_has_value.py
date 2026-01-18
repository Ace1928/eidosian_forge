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
def trait_has_value(self, name: str) -> bool:
    """Returns True if the specified trait has a value.

        This will return false even if ``getattr`` would return a
        dynamically generated default value. These default values
        will be recognized as existing only after they have been
        generated.

        Example

        .. code-block:: python

            class MyClass(HasTraits):
                i = Int()


            mc = MyClass()
            assert not mc.trait_has_value("i")
            mc.i  # generates a default value
            assert mc.trait_has_value("i")
        """
    return name in self._trait_values