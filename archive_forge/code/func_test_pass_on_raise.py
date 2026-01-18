import sys
from testtools import (
from testtools.matchers import (
def test_pass_on_raise(self):
    with ExpectedException(ValueError, 'tes.'):
        raise ValueError('test')