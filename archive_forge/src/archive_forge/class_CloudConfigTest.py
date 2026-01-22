from unittest import mock
import uuid
import yaml
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class CloudConfigTest(common.HeatTestCase):

    def setUp(self):
        super(CloudConfigTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.properties = {'cloud_config': {'foo': 'bar'}}
        self.stack = stack.Stack(self.ctx, 'software_config_test_stack', template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'config_mysql': {'Type': 'OS::Heat::CloudConfig', 'Properties': self.properties}}}))
        self.config = self.stack['config_mysql']
        self.rpc_client = mock.MagicMock()
        self.config._rpc_client = self.rpc_client

    def test_handle_create(self):
        config_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
        value = {'id': config_id}
        self.rpc_client.create_software_config.return_value = value
        self.config.id = 5
        self.config.uuid = uuid.uuid4().hex
        self.config.handle_create()
        self.assertEqual(config_id, self.config.resource_id)
        kwargs = self.rpc_client.create_software_config.call_args[1]
        self.assertEqual({'name': self.config.physical_resource_name(), 'config': '\n'.join(['#cloud-config', yaml.safe_dump({'foo': 'bar'})]), 'group': 'Heat::Ungrouped'}, kwargs)