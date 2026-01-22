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
class directional_link:
    """Link the trait of a source object with traits of target objects.

    Parameters
    ----------
    source : (object, attribute name) pair
    target : (object, attribute name) pair
    transform: callable (optional)
        Data transformation between source and target.

    Examples
    --------
    >>> class X(HasTraits):
    ...     value = Int()

    >>> src = X(value=1)
    >>> tgt = X(value=42)
    >>> c = directional_link((src, "value"), (tgt, "value"))

    Setting source updates target objects:
    >>> src.value = 5
    >>> tgt.value
    5

    Setting target does not update source object:
    >>> tgt.value = 6
    >>> src.value
    5

    """
    updating = False

    def __init__(self, source: t.Any, target: t.Any, transform: t.Any=None) -> None:
        self._transform = transform if transform else lambda x: x
        _validate_link(source, target)
        self.source, self.target = (source, target)
        self.link()

    def link(self) -> None:
        try:
            setattr(self.target[0], self.target[1], self._transform(getattr(self.source[0], self.source[1])))
        finally:
            self.source[0].observe(self._update, names=self.source[1])

    @contextlib.contextmanager
    def _busy_updating(self) -> t.Any:
        self.updating = True
        try:
            yield
        finally:
            self.updating = False

    def _update(self, change: t.Any) -> None:
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.target[0], self.target[1], self._transform(change.new))

    def unlink(self) -> None:
        self.source[0].unobserve(self._update, names=self.source[1])