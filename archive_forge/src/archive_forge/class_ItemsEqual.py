import collections
import logging
from unittest import mock
import fixtures
from oslotest import base
from testtools import compat
from testtools import matchers
from testtools import testcase
from taskflow import exceptions
from taskflow.tests import fixtures as taskflow_fixtures
from taskflow.tests import utils
from taskflow.utils import misc
class ItemsEqual(object):
    """Matches the items in two sequences.

    This matcher will validate that the provided sequence has the same elements
    as a reference sequence, regardless of the order.
    """

    def __init__(self, seq):
        self._seq = seq
        self._list = list(seq)

    def match(self, other):
        other_list = list(other)
        extra = misc.sequence_minus(other_list, self._list)
        missing = misc.sequence_minus(self._list, other_list)
        if extra or missing:
            msg = 'Sequences %s and %s do not have same items.' % (self._seq, other)
            if missing:
                msg += ' Extra items in first sequence: %s.' % missing
            if extra:
                msg += ' Extra items in second sequence: %s.' % extra
            return matchers.Mismatch(msg)
        return None