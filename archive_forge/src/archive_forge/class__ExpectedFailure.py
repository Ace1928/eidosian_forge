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
class _ExpectedFailure(Exception):
    """An expected failure occurred.

    Note that this exception is private plumbing in testtools' testcase
    module.
    """