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
def assertThat(self, matchee, matcher, message='', verbose=False):
    """Assert that matchee is matched by matcher.

        :param matchee: An object to match with matcher.
        :param matcher: An object meeting the testtools.Matcher protocol.
        :raises MismatchError: When matcher does not match thing.
        """
    mismatch_error = self._matchHelper(matchee, matcher, message, verbose)
    if mismatch_error is not None:
        raise mismatch_error