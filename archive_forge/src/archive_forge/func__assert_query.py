import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def _assert_query(url, expected):
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(parts.query)
    assert query == expected