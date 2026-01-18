import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_update_query_existing_params():
    uri = 'http://www.google.com?x=y'
    updated = _helpers.update_query(uri, {'a': 'b', 'c': 'd&'})
    _assert_query(updated, {'x': ['y'], 'a': ['b'], 'c': ['d&']})