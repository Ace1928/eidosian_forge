import sys
from testtools import (
from testtools.matchers import (
def test_raise_on_general_mismatch(self):
    matcher = AfterPreprocessing(str, Equals('test'))
    value_error = ValueError('mismatch')
    try:
        with ExpectedException(ValueError, matcher):
            raise value_error
    except AssertionError:
        e = sys.exc_info()[1]
        self.assertEqual(matcher.match(value_error).describe(), str(e))
    else:
        self.fail('AssertionError not raised.')