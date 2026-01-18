from unittest import mock
import uuid
import stevedore
from keystone.api._shared import authentication
from keystone import auth
from keystone.auth.plugins import base
from keystone.auth.plugins import mapped
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import auth_plugins
def test_unsupported_auth_method(self):
    method_name = uuid.uuid4().hex
    auth_data = {'methods': [method_name]}
    auth_data[method_name] = {'test': 'test'}
    auth_data = {'identity': auth_data}
    self.assertRaises(exception.AuthMethodNotSupported, auth.core.AuthInfo.create, auth_data)