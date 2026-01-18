import sys
from testtools import (
from testtools.matchers import (
def test_raise_if_no_exception(self):
    try:
        with ExpectedException(TypeError, 'tes.'):
            pass
    except AssertionError:
        e = sys.exc_info()[1]
        self.assertEqual('TypeError not raised.', str(e))
    else:
        self.fail('AssertionError not raised.')