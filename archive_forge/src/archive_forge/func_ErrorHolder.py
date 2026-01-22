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
def ErrorHolder(test_id, error, short_description=None, details=None):
    """Construct an `ErrorHolder`.

    :param test_id: The id of the test.
    :param error: The exc info tuple that will be used as the test's error.
        This is inserted into the details as 'traceback' - any existing key
        will be overridden.
    :param short_description: An optional short description of the test.
    :param details: Outcome details as accepted by addSuccess etc.
    """
    return PlaceHolder(test_id, short_description=short_description, details=details, outcome='addError', error=error)