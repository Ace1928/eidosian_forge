import copy
import json
import time
import unittest
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.exceptions import ClientException
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import base as v3_base
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_with_multiple_scopes(self):
    s = session.Session()
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS, domain_id='x', project_id='x')
    self.assertRaises(exceptions.AuthorizationFailure, a.get_auth_ref, s)
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS, domain_id='x', trust_id='x')
    self.assertRaises(exceptions.AuthorizationFailure, a.get_auth_ref, s)