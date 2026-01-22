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
def _test_mapped_invocation_with_method_name(self, method_name):
    with mock.patch.object(auth.plugins.mapped.Mapped, 'authenticate', return_value=None) as authenticate:
        auth_data = {'identity': {'methods': [method_name], method_name: {'protocol': method_name}}}
        auth_info = auth.core.AuthInfo.create(auth_data)
        auth_context = auth.core.AuthContext(method_names=[], user_id=uuid.uuid4().hex)
        with self.make_request():
            authentication.authenticate(auth_info, auth_context)
        (auth_payload,), kwargs = authenticate.call_args
        self.assertEqual(method_name, auth_payload['protocol'])