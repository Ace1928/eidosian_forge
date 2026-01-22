from unittest import mock
from heat.common import exception
from heat.engine.resources.openstack.heat import structured_config as sc
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import stack as parser
from heat.engine import template
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
class StructuredDeploymentDerivedTest(common.HeatTestCase):
    template = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'deploy_mysql': {'Type': 'OS::Heat::StructuredDeployment'}}}

    def setUp(self):
        super(StructuredDeploymentDerivedTest, self).setUp()
        self.ctx = utils.dummy_context()
        props = {'server': '9f1f0e00-05d2-4ca5-8602-95021f19c9d0', 'input_values': {'bar': 'baz'}}
        self.template['Resources']['deploy_mysql']['Properties'] = props
        self.stack = parser.Stack(self.ctx, 'software_deploly_test_stack', template.Template(self.template))
        self.deployment = self.stack['deploy_mysql']

    def test_build_derived_config(self):
        source = {'config': {'foo': {'get_input': 'bar'}}}
        inputs = [swc_io.InputConfig(name='bar', value='baz')]
        result = self.deployment._build_derived_config('CREATE', source, inputs, {})
        self.assertEqual({'foo': 'baz'}, result)

    def test_build_derived_config_params_with_empty_config(self):
        source = {}
        source[rpc_api.SOFTWARE_CONFIG_INPUTS] = []
        source[rpc_api.SOFTWARE_CONFIG_OUTPUTS] = []
        result = self.deployment._build_derived_config_params('CREATE', source)
        self.assertEqual('Heat::Ungrouped', result['group'])
        self.assertEqual({}, result['config'])
        self.assertEqual(self.deployment.physical_resource_name(), result['name'])
        self.assertIn({'name': 'bar', 'type': 'String', 'value': 'baz'}, result['inputs'])
        self.assertIsNone(result['options'])
        self.assertEqual([], result['outputs'])