from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
class KeystoneRoleConstraintTest(common.HeatTestCase):

    def test_expected_exceptions(self):
        self.assertEqual((exception.EntityNotFound,), ks_constr.KeystoneRoleConstraint.expected_exceptions, 'KeystoneRoleConstraint expected exceptions error')

    def test_constraint(self):
        constraint = ks_constr.KeystoneRoleConstraint()
        client_mock = mock.MagicMock()
        client_plugin_mock = mock.MagicMock()
        client_plugin_mock.get_role_id.return_value = None
        client_mock.client_plugin.return_value = client_plugin_mock
        self.assertIsNone(constraint.validate_with_client(client_mock, 'role_1'))
        self.assertRaises(exception.EntityNotFound, constraint.validate_with_client, client_mock, '')
        client_plugin_mock.get_role_id.assert_called_once_with('role_1')