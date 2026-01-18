import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_update_query_replace_param():
    base_uri = 'http://www.google.com'
    uri = base_uri + '?x=a'
    updated = _helpers.update_query(uri, {'x': 'b', 'y': 'c'})
    _assert_query(updated, {'x': ['b'], 'y': ['c']})