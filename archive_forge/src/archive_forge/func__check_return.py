import abc
import sys
import traceback
import warnings
from io import StringIO
from decorator import decorator
from traitlets.config.configurable import Configurable
from .getipython import get_ipython
from ..utils.sentinel import Sentinel
from ..utils.dir2 import get_real_method
from ..lib import pretty
from traitlets import (
from typing import Any
def _check_return(self, r, obj):
    r = super(MimeBundleFormatter, self)._check_return(r, obj)
    if r is None:
        return ({}, {})
    if not isinstance(r, tuple):
        return (r, {})
    return r