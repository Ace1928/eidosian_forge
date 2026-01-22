import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import (
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import (
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
def _collectWarnings(observeWarning, f, *args, **kwargs):
    """
    Call C{f} with C{args} positional arguments and C{kwargs} keyword arguments
    and collect all warnings which are emitted as a result in a list.

    @param observeWarning: A callable which will be invoked with a L{_Warning}
        instance each time a warning is emitted.

    @return: The return value of C{f(*args, **kwargs)}.
    """

    def showWarning(message, category, filename, lineno, file=None, line=None):
        assert isinstance(message, Warning)
        observeWarning(_Warning(str(message), category, filename, lineno))
    _setWarningRegistryToNone(sys.modules)
    origFilters = warnings.filters[:]
    origShow = warnings.showwarning
    warnings.simplefilter('always')
    try:
        warnings.showwarning = showWarning
        result = f(*args, **kwargs)
    finally:
        warnings.filters[:] = origFilters
        warnings.showwarning = origShow
    return result