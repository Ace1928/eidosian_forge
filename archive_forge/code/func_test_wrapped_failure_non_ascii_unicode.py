import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_wrapped_failure_non_ascii_unicode(self):
    hi_cn = u'嗨'
    fail = ValueError(hi_cn)
    self.assertEqual(hi_cn, encodeutils.exception_to_unicode(fail))
    fail = failure.Failure.from_exception(fail)
    wrapped_fail = exceptions.WrappedFailure([fail])
    expected_result = u'WrappedFailure: [Failure: ValueError: %s]' % hi_cn
    self.assertEqual(expected_result, str(wrapped_fail))