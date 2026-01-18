import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy
def wrapper_setattr(self, name, value):
    """Actual `self.setattr` rather than self._wrapped.setattr"""
    return object.__setattr__(self, name, value)