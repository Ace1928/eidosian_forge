import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy
def wrapper_getattr(self, name):
    """Actual `self.getattr` rather than self._wrapped.getattr"""
    try:
        return object.__getattr__(self, name)
    except AttributeError:
        return getattr(self, name)