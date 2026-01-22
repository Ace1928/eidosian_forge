import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
class ResourceChainTest(common.HeatTestCase):

    def setUp(self):
        super(ResourceChainTest, self).setUp()
        self.stack = None

    def test_child_template_without_concurrency(self):
        chain = self._create_chain(TEMPLATE)
        child_template = chain.child_template()
        tmpl = child_template.t
        self.assertEqual('2015-04-30', tmpl['heat_template_version'])
        self.assertEqual(2, len(child_template.t['resources']))
        resource = tmpl['resources']['0']
        self.assertEqual('OS::Heat::SoftwareConfig', resource['type'])
        self.assertEqual(RESOURCE_PROPERTIES, resource['properties'])
        self.assertNotIn('depends_on', resource)
        resource = tmpl['resources']['1']
        self.assertEqual('OS::Heat::StructuredConfig', resource['type'])
        self.assertEqual(RESOURCE_PROPERTIES, resource['properties'])
        self.assertEqual(['0'], resource['depends_on'])

    @mock.patch.object(service_objects.Service, 'active_service_count')
    def test_child_template_with_concurrent(self, mock_count):
        tmpl_def = copy.deepcopy(TEMPLATE)
        tmpl_def['resources']['test-chain']['properties']['concurrent'] = True
        chain = self._create_chain(tmpl_def)
        mock_count.return_value = 5
        child_template = chain.child_template()
        tmpl = child_template.t
        resource = tmpl['resources']['0']
        self.assertNotIn('depends_on', resource)
        resource = tmpl['resources']['1']
        self.assertNotIn('depends_on', resource)

    @mock.patch.object(service_objects.Service, 'active_service_count')
    def test_child_template_with_concurrent_limit(self, mock_count):
        tmpl_def = copy.deepcopy(TEMPLATE)
        tmpl_def['resources']['test-chain']['properties']['concurrent'] = True
        tmpl_def['resources']['test-chain']['properties']['resources'] = ['OS::Heat::SoftwareConfig', 'OS::Heat::StructuredConfig', 'OS::Heat::SoftwareConfig', 'OS::Heat::StructuredConfig']
        chain = self._create_chain(tmpl_def)
        mock_count.return_value = 2
        child_template = chain.child_template()
        tmpl = child_template.t
        resource = tmpl['resources']['0']
        self.assertNotIn('depends_on', resource)
        resource = tmpl['resources']['1']
        self.assertNotIn('depends_on', resource)
        resource = tmpl['resources']['2']
        self.assertEqual(['0'], resource['depends_on'])
        resource = tmpl['resources']['3']
        self.assertEqual(['1'], resource['depends_on'])

    def test_child_template_default_concurrent(self):
        tmpl_def = copy.deepcopy(TEMPLATE)
        tmpl_def['resources']['test-chain']['properties'].pop('concurrent')
        chain = self._create_chain(tmpl_def)
        child_template = chain.child_template()
        tmpl = child_template.t
        resource = tmpl['resources']['0']
        self.assertNotIn('depends_on', resource)
        resource = tmpl['resources']['1']
        self.assertEqual(['0'], resource['depends_on'])

    def test_child_template_empty_resource_list(self):
        tmpl_def = copy.deepcopy(TEMPLATE)
        tmpl_def['resources']['test-chain']['properties']['resources'] = []
        chain = self._create_chain(tmpl_def)
        child_template = chain.child_template()
        tmpl = child_template.t
        self.assertNotIn('resources', tmpl)
        self.assertIn('heat_template_version', tmpl)

    def test_validate_nested_stack(self):
        chain = self._create_chain(TEMPLATE)
        chain.validate_nested_stack()

    def test_validate_reference_attr_with_none_ref(self):
        chain = self._create_chain(TEMPLATE)
        self.patchobject(chain, 'referenced_attrs', return_value=set([('config', None)]))
        self.assertIsNone(chain.validate())

    def test_validate_incompatible_properties(self):
        tmpl_def = copy.deepcopy(TEMPLATE)
        tmpl_res_prop = tmpl_def['resources']['test-chain']['properties']
        res_list = tmpl_res_prop['resources']
        res_list.append('OS::Heat::RandomString')
        chain = self._create_chain(tmpl_def)
        try:
            chain.validate_nested_stack()
            self.fail('Exception expected')
        except exception.StackValidationFailed as e:
            self.assertEqual('property error: resources.test<nested_stack>.resources[2].properties: unknown property group', e.message.lower())

    def test_validate_fake_resource_type(self):
        tmpl_def = copy.deepcopy(TEMPLATE)
        tmpl_res_prop = tmpl_def['resources']['test-chain']['properties']
        res_list = tmpl_res_prop['resources']
        res_list.append('foo')
        chain = self._create_chain(tmpl_def)
        try:
            chain.validate_nested_stack()
            self.fail('Exception expected')
        except exception.StackValidationFailed as e:
            self.assertIn('could not be found', e.message.lower())
            self.assertIn('foo', e.message)

    @mock.patch.object(resource_chain.ResourceChain, 'create_with_template')
    def test_handle_create(self, mock_create):
        chain = self._create_chain(TEMPLATE)
        chain.handle_create()
        expected_tmpl = chain.child_template()
        mock_create.assert_called_once_with(expected_tmpl)

    @mock.patch.object(resource_chain.ResourceChain, 'update_with_template')
    def test_handle_update(self, mock_update):
        chain = self._create_chain(TEMPLATE)
        json_snippet = rsrc_defn.ResourceDefinition('test-chain', 'OS::Heat::ResourceChain', TEMPLATE['resources']['test-chain']['properties'])
        chain.handle_update(json_snippet, None, None)
        expected_tmpl = chain.child_template()
        mock_update.assert_called_once_with(expected_tmpl)

    def test_child_params(self):
        chain = self._create_chain(TEMPLATE)
        self.assertEqual({}, chain.child_params())

    def _create_chain(self, t):
        self.stack = utils.parse_stack(t)
        snip = self.stack.t.resource_definitions(self.stack)['test-chain']
        chain = resource_chain.ResourceChain('test', snip, self.stack)
        return chain

    def test_get_attribute_convg(self):
        cache_data = {'test-chain': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'attrs': {'refs': ['rsrc1', 'rsrc2']}})}
        stack = utils.parse_stack(TEMPLATE, cache_data=cache_data)
        rsrc = stack.defn['test-chain']
        self.assertEqual(['rsrc1', 'rsrc2'], rsrc.FnGetAtt('refs'))