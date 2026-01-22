import copy
import functools
import itertools
import sys
import types
import unittest
import warnings
from testtools.compat import reraise
from testtools import content
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.matchers._basic import _FlippedEquals
from testtools.monkey import patch
from testtools.runtest import (
from testtools.testresult import (
def _formatTypes(self, classOrIterable):
    """Format a class or a bunch of classes for display in an error."""
    className = getattr(classOrIterable, '__name__', None)
    if className is None:
        className = ', '.join((klass.__name__ for klass in classOrIterable))
    return className