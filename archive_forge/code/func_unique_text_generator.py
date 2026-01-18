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
def unique_text_generator(prefix):
    """Generates unique text values.

    Generates text values that are unique. Use this when you need arbitrary
    text in your test, or as a helper for custom anonymous factory methods.

    :param prefix: The prefix for text.
    :return: text that looks like '<prefix>-<text_with_unicode>'.
    :rtype: str
    """
    BASE_CP = 7680
    CP_RANGE = 7936 - BASE_CP
    index = 0
    while True:
        unique_text = _unique_text(BASE_CP, CP_RANGE, index)
        yield f'{prefix}-{unique_text}'
        index = index + 1