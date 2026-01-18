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
def with_properties(self, **kwargs: typing.Any) -> KeyringBackend:
    alt = copy.copy(self)
    vars(alt).update(kwargs)
    return alt