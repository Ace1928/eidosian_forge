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
def _update_target(self, change: t.Any) -> None:
    if self.updating:
        return
    with self._busy_updating():
        setattr(self.target[0], self.target[1], self._transform(change.new))
        if getattr(self.source[0], self.source[1]) != change.new:
            raise TraitError(f'Broken link {self}: the source value changed while updating the target.')