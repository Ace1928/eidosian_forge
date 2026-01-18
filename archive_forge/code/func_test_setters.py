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
def test_setters(self):
    availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
    access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
    rules = [access_boundary_rule]
    other_availability_condition = make_availability_condition(OTHER_EXPRESSION, OTHER_TITLE, OTHER_DESCRIPTION)
    other_access_boundary_rule = make_access_boundary_rule(OTHER_AVAILABLE_RESOURCE, OTHER_AVAILABLE_PERMISSIONS, other_availability_condition)
    other_rules = [other_access_boundary_rule]
    credential_access_boundary = make_credential_access_boundary(rules)
    credential_access_boundary.rules = other_rules
    assert credential_access_boundary.rules == tuple(other_rules)