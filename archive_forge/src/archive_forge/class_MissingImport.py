from __future__ import (absolute_import, division, print_function)
from functools import reduce
import inspect
import math
import operator
import sys
from pkg_resources import parse_requirements, parse_version
import numpy as np
import pytest
class MissingImport(object):

    def __init__(self, modname, exc):
        self._modname = modname
        self._exc = exc

    def __getattribute__(self, attr):
        if attr in ('_modname', '_exc'):
            return object.__getattribute__(self, attr)
        else:
            raise self._exc

    def __getitem__(self, key):
        raise self._exc

    def __call__(self, *args, **kwargs):
        raise self._exc