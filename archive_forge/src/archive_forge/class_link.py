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
class link:
    """Link traits from different objects together so they remain in sync.

    Parameters
    ----------
    source : (object / attribute name) pair
    target : (object / attribute name) pair
    transform: iterable with two callables (optional)
        Data transformation between source and target and target and source.

    Examples
    --------
    >>> class X(HasTraits):
    ...     value = Int()

    >>> src = X(value=1)
    >>> tgt = X(value=42)
    >>> c = link((src, "value"), (tgt, "value"))

    Setting source updates target objects:
    >>> src.value = 5
    >>> tgt.value
    5
    """
    updating = False

    def __init__(self, source: t.Any, target: t.Any, transform: t.Any=None) -> None:
        _validate_link(source, target)
        self.source, self.target = (source, target)
        self._transform, self._transform_inv = transform if transform else (lambda x: x,) * 2
        self.link()

    def link(self) -> None:
        try:
            setattr(self.target[0], self.target[1], self._transform(getattr(self.source[0], self.source[1])))
        finally:
            self.source[0].observe(self._update_target, names=self.source[1])
            self.target[0].observe(self._update_source, names=self.target[1])

    @contextlib.contextmanager
    def _busy_updating(self) -> t.Any:
        self.updating = True
        try:
            yield
        finally:
            self.updating = False

    def _update_target(self, change: t.Any) -> None:
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.target[0], self.target[1], self._transform(change.new))
            if getattr(self.source[0], self.source[1]) != change.new:
                raise TraitError(f'Broken link {self}: the source value changed while updating the target.')

    def _update_source(self, change: t.Any) -> None:
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.source[0], self.source[1], self._transform_inv(change.new))
            if getattr(self.target[0], self.target[1]) != change.new:
                raise TraitError(f'Broken link {self}: the target value changed while updating the source.')

    def unlink(self) -> None:
        self.source[0].unobserve(self._update_target, names=self.source[1])
        self.target[0].unobserve(self._update_source, names=self.target[1])