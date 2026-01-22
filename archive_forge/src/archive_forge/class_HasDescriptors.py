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
class HasDescriptors(metaclass=MetaHasDescriptors):
    """The base class for all classes that have descriptors."""

    def __new__(*args: t.Any, **kwargs: t.Any) -> t.Any:
        cls = args[0]
        args = args[1:]
        new_meth = super(HasDescriptors, cls).__new__
        if new_meth is object.__new__:
            inst = new_meth(cls)
        else:
            inst = new_meth(cls, *args, **kwargs)
        inst.setup_instance(*args, **kwargs)
        return inst

    def setup_instance(*args: t.Any, **kwargs: t.Any) -> None:
        """
        This is called **before** self.__init__ is called.
        """
        self = args[0]
        args = args[1:]
        self._cross_validation_lock = False
        cls = self.__class__
        for init in cls._instance_inits:
            init(self)