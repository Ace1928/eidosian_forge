import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
class FromExceptionTestCase(test.TestCase, GeneralFailureObjTestsMixin):

    def setUp(self):
        super(FromExceptionTestCase, self).setUp()
        self.fail_obj = failure.Failure.from_exception(RuntimeError('Woot!'))

    def test_pformat_no_traceback(self):
        text = self.fail_obj.pformat(traceback=True)
        self.assertIn('Traceback not available', text)