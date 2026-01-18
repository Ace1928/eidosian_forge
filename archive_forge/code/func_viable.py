from __future__ import annotations
import os
import abc
import logging
import operator
import copy
import typing
from .py312compat import metadata
from . import credentials, errors, util
from ._compat import properties
@properties.classproperty
def viable(cls):
    with errors.ExceptionRaisedContext() as exc:
        cls.priority
    return not exc