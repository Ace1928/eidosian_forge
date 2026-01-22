import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
class WrappedFailureTestCase(test.TestCase):

    def test_simple_iter(self):
        fail_obj = _captured_failure('Woot!')
        wf = exceptions.WrappedFailure([fail_obj])
        self.assertEqual(1, len(wf))
        self.assertEqual([fail_obj], list(wf))

    def test_simple_check(self):
        fail_obj = _captured_failure('Woot!')
        wf = exceptions.WrappedFailure([fail_obj])
        self.assertEqual(RuntimeError, wf.check(RuntimeError))
        self.assertIsNone(wf.check(ValueError))

    def test_two_failures(self):
        fls = [_captured_failure('Woot!'), _captured_failure('Oh, not again!')]
        wf = exceptions.WrappedFailure(fls)
        self.assertEqual(2, len(wf))
        self.assertEqual(fls, list(wf))

    def test_flattening(self):
        f1 = _captured_failure('Wrap me')
        f2 = _captured_failure('Wrap me, too')
        f3 = _captured_failure('Woot!')
        try:
            raise exceptions.WrappedFailure([f1, f2])
        except Exception:
            fail_obj = failure.Failure()
        wf = exceptions.WrappedFailure([fail_obj, f3])
        self.assertEqual([f1, f2, f3], list(wf))