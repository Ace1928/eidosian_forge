import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_none_ne_tuple(self):
    exc_info = _make_exc_info('Woot!')
    self.assertFalse(failure._are_equal_exc_info_tuples(None, exc_info))