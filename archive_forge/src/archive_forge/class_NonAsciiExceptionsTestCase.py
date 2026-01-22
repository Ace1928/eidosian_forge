import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
class NonAsciiExceptionsTestCase(test.TestCase):

    def test_exception_with_non_ascii_str(self):
        bad_string = chr(200)
        excp = ValueError(bad_string)
        fail = failure.Failure.from_exception(excp)
        self.assertEqual(encodeutils.exception_to_unicode(excp), fail.exception_str)
        expected = u'Failure: ValueError: È'
        self.assertEqual(expected, str(fail))

    def test_exception_non_ascii_unicode(self):
        hi_ru = u'привет'
        fail = failure.Failure.from_exception(ValueError(hi_ru))
        self.assertEqual(hi_ru, fail.exception_str)
        self.assertIsInstance(fail.exception_str, str)
        self.assertEqual(u'Failure: ValueError: %s' % hi_ru, str(fail))

    def test_wrapped_failure_non_ascii_unicode(self):
        hi_cn = u'嗨'
        fail = ValueError(hi_cn)
        self.assertEqual(hi_cn, encodeutils.exception_to_unicode(fail))
        fail = failure.Failure.from_exception(fail)
        wrapped_fail = exceptions.WrappedFailure([fail])
        expected_result = u'WrappedFailure: [Failure: ValueError: %s]' % hi_cn
        self.assertEqual(expected_result, str(wrapped_fail))

    def test_failure_equality_with_non_ascii_str(self):
        bad_string = chr(200)
        fail = failure.Failure.from_exception(ValueError(bad_string))
        copied = fail.copy()
        self.assertEqual(fail, copied)

    def test_failure_equality_non_ascii_unicode(self):
        hi_ru = u'привет'
        fail = failure.Failure.from_exception(ValueError(hi_ru))
        copied = fail.copy()
        self.assertEqual(fail, copied)