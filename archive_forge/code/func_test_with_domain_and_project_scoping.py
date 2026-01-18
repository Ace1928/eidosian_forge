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
def test_with_domain_and_project_scoping(self):
    a = v3.Password(self.TEST_URL, username='username', password='password', project_id='project', domain_id='domain')
    self.assertTrue(a.has_scope_parameters)
    self.assertRaises(exceptions.AuthorizationFailure, a.get_token, None)
    self.assertRaises(exceptions.AuthorizationFailure, a.get_headers, None)