import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_unpadded_urlsafe_b64encode():
    cases = [(b'', b''), (b'a', b'YQ'), (b'aa', b'YWE'), (b'aaa', b'YWFh')]
    for case, expected in cases:
        assert _helpers.unpadded_urlsafe_b64encode(case) == expected