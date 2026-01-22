from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
class TestInstanceShow(TestInstances):

    def setUp(self):
        super(TestInstanceShow, self).setUp()
        self.cmd = database_instances.ShowDatabaseInstance(self.app, None)
        self.columns = ('addresses', 'allowed_cidrs', 'datastore', 'datastore_version', 'datastore_version_number', 'flavor', 'id', 'name', 'operating_status', 'public', 'region', 'replica_of', 'status', 'tenant_id', 'volume')

    def test_show(self):
        instance_id = self.random_uuid()
        name = self.random_name('test-show')
        flavor_id = self.random_uuid()
        primary_id = self.random_uuid()
        tenant_id = self.random_uuid()
        inst = {'id': instance_id, 'name': name, 'status': 'ACTIVE', 'operating_status': 'HEALTHY', 'addresses': [{'type': 'private', 'address': '10.0.0.13'}], 'volume': {'size': 2}, 'flavor': {'id': flavor_id}, 'region': 'regionOne', 'datastore': {'version': '5.7.29', 'type': 'mysql', 'version_number': '5.7.29'}, 'tenant_id': tenant_id, 'replica_of': {'id': primary_id}, 'access': {'is_public': False, 'allowed_cidrs': []}}
        self.instance_client.get.return_value = instances.Instance(mock.MagicMock(), inst)
        parsed_args = self.check_parser(self.cmd, [instance_id], [])
        columns, data = self.cmd.take_action(parsed_args)
        values = ([{'address': '10.0.0.13', 'type': 'private'}], [], 'mysql', '5.7.29', '5.7.29', flavor_id, instance_id, name, 'HEALTHY', False, 'regionOne', primary_id, 'ACTIVE', tenant_id, 2)
        self.assertEqual(self.columns, columns)
        self.assertEqual(values, data)