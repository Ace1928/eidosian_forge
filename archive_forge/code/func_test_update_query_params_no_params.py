import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_update_query_params_no_params():
    uri = 'http://www.google.com'
    updated = _helpers.update_query(uri, {'a': 'b'})
    assert updated == uri + '?a=b'