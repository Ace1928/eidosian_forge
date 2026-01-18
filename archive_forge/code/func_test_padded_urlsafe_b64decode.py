import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_padded_urlsafe_b64decode():
    cases = [('YQ==', b'a'), ('YQ', b'a'), ('YWE=', b'aa'), ('YWE', b'aa'), ('YWFhYQ==', b'aaaa'), ('YWFhYQ', b'aaaa'), ('YWFhYWE=', b'aaaaa'), ('YWFhYWE', b'aaaaa')]
    for case, expected in cases:
        assert _helpers.padded_urlsafe_b64decode(case) == expected