import sys
from testtools import (
from testtools.matchers import (
def test_raise_on_error_mismatch(self):
    try:
        with ExpectedException(TypeError, 'tes.'):
            raise ValueError('mismatch')
    except ValueError:
        e = sys.exc_info()[1]
        self.assertEqual('mismatch', str(e))
    else:
        self.fail('ValueError not raised.')