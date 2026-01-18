import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_utcnow():
    assert isinstance(_helpers.utcnow(), datetime.datetime)