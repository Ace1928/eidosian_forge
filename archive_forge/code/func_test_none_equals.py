import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_none_equals(self):
    self.assertTrue(failure._are_equal_exc_info_tuples(None, None))