import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_simple_check(self):
    fail_obj = _captured_failure('Woot!')
    wf = exceptions.WrappedFailure([fail_obj])
    self.assertEqual(RuntimeError, wf.check(RuntimeError))
    self.assertIsNone(wf.check(ValueError))