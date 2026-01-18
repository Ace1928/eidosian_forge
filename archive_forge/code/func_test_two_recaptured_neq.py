import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_two_recaptured_neq(self):
    captured = _captured_failure('Woot!')
    fail_obj = failure.Failure(exception_str=captured.exception_str, traceback_str=captured.traceback_str, exc_type_names=list(captured))
    new_exc_str = captured.exception_str.replace('Woot', 'w00t')
    fail_obj2 = failure.Failure(exception_str=new_exc_str, traceback_str=captured.traceback_str, exc_type_names=list(captured))
    self.assertNotEqual(fail_obj, fail_obj2)
    self.assertFalse(fail_obj2.matches(fail_obj))