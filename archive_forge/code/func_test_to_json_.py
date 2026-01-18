import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import credentials
from google.auth import downscoped
from google.auth import exceptions
from google.auth import transport
def test_to_json_(self):
    availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
    assert availability_condition.to_json() == {'expression': EXPRESSION, 'title': TITLE, 'description': DESCRIPTION}