import copy
from unittest import mock
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine.cfn import functions as cfn_functions
from heat.engine.cfn import parameters as cfn_param
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import parameters as hot_param
from heat.engine.hot import template as hot_template
from heat.engine import resource
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class HotStackTest(common.HeatTestCase):
    """Test stack function when stack was created from HOT template."""

    def setUp(self):
        super(HotStackTest, self).setUp()
        self.tmpl = template.Template(copy.deepcopy(empty_template))
        self.ctx = utils.dummy_context()

    def resolve(self, snippet):
        return function.resolve(self.stack.t.parse(self.stack.defn, snippet))

    def test_repeat_get_attr(self):
        """Test repeat function with get_attr function as an argument."""
        tmpl = template.Template(hot_tpl_complex_attrs_all_attrs)
        self.stack = parser.Stack(self.ctx, 'test_repeat_get_attr', tmpl)
        snippet = {'repeat': {'template': 'this is %var%', 'for_each': {'%var%': {'get_attr': ['resource1', 'list']}}}}
        repeat = self.stack.t.parse(self.stack.defn, snippet)
        self.stack.store()
        with mock.patch.object(rsrc_defn.ResourceDefinition, 'dep_attrs') as mock_da:
            mock_da.return_value = ['list']
            self.stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
        self.assertEqual(['this is foo', 'this is bar'], function.resolve(repeat))

    def test_get_attr_multiple_rsrc_status(self):
        """Test resolution of get_attr occurrences in HOT template."""
        hot_tpl = hot_tpl_generic_resource
        self.stack = parser.Stack(self.ctx, 'test_get_attr', template.Template(hot_tpl))
        self.stack.store()
        with mock.patch.object(rsrc_defn.ResourceDefinition, 'dep_attrs') as mock_da:
            mock_da.return_value = ['foo']
            self.stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
        snippet = {'Value': {'get_attr': ['resource1', 'foo']}}
        rsrc = self.stack['resource1']
        for action, status in ((rsrc.CREATE, rsrc.IN_PROGRESS), (rsrc.CREATE, rsrc.COMPLETE), (rsrc.RESUME, rsrc.IN_PROGRESS), (rsrc.RESUME, rsrc.COMPLETE), (rsrc.UPDATE, rsrc.IN_PROGRESS), (rsrc.UPDATE, rsrc.COMPLETE)):
            rsrc.state_set(action, status)
            self.assertEqual({'Value': 'resource1'}, self.resolve(snippet))

    def test_get_attr_invalid(self):
        """Test resolution of get_attr occurrences in HOT template."""
        hot_tpl = hot_tpl_generic_resource
        self.stack = parser.Stack(self.ctx, 'test_get_attr', template.Template(hot_tpl))
        self.stack.store()
        self.stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
        self.assertRaises(exception.InvalidTemplateAttribute, self.resolve, {'Value': {'get_attr': ['resource1', 'NotThere']}})

    def test_get_attr_invalid_resource(self):
        """Test resolution of get_attr occurrences in HOT template."""
        hot_tpl = hot_tpl_complex_attrs
        self.stack = parser.Stack(self.ctx, 'test_get_attr_invalid_none', template.Template(hot_tpl))
        self.stack.store()
        self.stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
        snippet = {'Value': {'get_attr': ['resource2', 'who_cares']}}
        self.assertRaises(exception.InvalidTemplateReference, self.resolve, snippet)

    def test_get_resource(self):
        """Test resolution of get_resource occurrences in HOT template."""
        hot_tpl = hot_tpl_generic_resource
        self.stack = parser.Stack(self.ctx, 'test_get_resource', template.Template(hot_tpl))
        self.stack.store()
        self.stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
        snippet = {'value': {'get_resource': 'resource1'}}
        self.assertEqual({'value': 'resource1'}, self.resolve(snippet))

    def test_set_param_id(self):
        tmpl = template.Template(hot_tpl_empty)
        self.stack = parser.Stack(self.ctx, 'param_id_test', tmpl)
        self.assertEqual('None', self.stack.parameters['OS::stack_id'])
        self.stack.store()
        stack_identifier = self.stack.identifier()
        self.assertEqual(self.stack.id, self.stack.parameters['OS::stack_id'])
        self.assertEqual(stack_identifier.stack_id, self.stack.parameters['OS::stack_id'])

    def test_set_wrong_param(self):
        tmpl = template.Template(hot_tpl_empty)
        stack_id = identifier.HeatIdentifier('', 'stack_testit', None)
        params = tmpl.parameters(None, {})
        self.assertFalse(params.set_stack_id(None))
        self.assertTrue(params.set_stack_id(stack_id))

    def test_set_param_id_update(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'resources': {'AResource': {'type': 'ResourceWithPropsType', 'metadata': {'Bar': {'get_param': 'OS::stack_id'}}, 'properties': {'Foo': 'abc'}}}})
        self.stack = parser.Stack(self.ctx, 'update_stack_id_test', tmpl)
        self.stack.store()
        self.stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
        stack_id = self.stack.parameters['OS::stack_id']
        tmpl2 = template.Template({'heat_template_version': '2013-05-23', 'resources': {'AResource': {'type': 'ResourceWithPropsType', 'metadata': {'Bar': {'get_param': 'OS::stack_id'}}, 'properties': {'Foo': 'xyz'}}}})
        updated_stack = parser.Stack(self.ctx, 'updated_stack', tmpl2)
        self.stack.update(updated_stack)
        self.assertEqual((parser.Stack.UPDATE, parser.Stack.COMPLETE), self.stack.state)
        self.assertEqual('xyz', self.stack['AResource'].properties['Foo'])
        self.assertEqual(stack_id, self.stack['AResource'].metadata_get()['Bar'])

    def test_load_param_id(self):
        tmpl = template.Template(hot_tpl_empty)
        self.stack = parser.Stack(self.ctx, 'param_load_id_test', tmpl)
        self.stack.store()
        stack_identifier = self.stack.identifier()
        self.assertEqual(stack_identifier.stack_id, self.stack.parameters['OS::stack_id'])
        newstack = parser.Stack.load(self.ctx, stack_id=self.stack.id)
        self.assertEqual(stack_identifier.stack_id, newstack.parameters['OS::stack_id'])

    def test_update_modify_param_ok_replace(self):
        tmpl = {'heat_template_version': '2013-05-23', 'parameters': {'foo': {'type': 'string'}}, 'resources': {'AResource': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'get_param': 'foo'}}}}}
        self.stack = parser.Stack(self.ctx, 'update_test_stack', template.Template(tmpl, env=environment.Environment({'foo': 'abc'})))
        self.stack.store()
        self.stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
        updated_stack = parser.Stack(self.ctx, 'updated_stack', template.Template(tmpl, env=environment.Environment({'foo': 'xyz'})))

        def check_props_and_raise(*args):
            self.assertEqual('abc', self.stack['AResource'].properties['Foo'])
            raise resource.UpdateReplace()
        mock_update = self.patchobject(generic_rsrc.ResourceWithProps, 'update_template_diff', side_effect=check_props_and_raise)
        self.stack.update(updated_stack)
        self.assertEqual((parser.Stack.UPDATE, parser.Stack.COMPLETE), self.stack.state)
        self.assertEqual('xyz', self.stack['AResource'].properties['Foo'])
        mock_update.assert_called_once_with(rsrc_defn.ResourceDefinition('AResource', 'ResourceWithPropsType', properties={'Foo': 'xyz'}), rsrc_defn.ResourceDefinition('AResource', 'ResourceWithPropsType', properties={'Foo': 'abc'}))

    def test_update_modify_files_ok_replace(self):
        tmpl = {'heat_template_version': '2013-05-23', 'parameters': {}, 'resources': {'AResource': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'get_file': 'foo'}}}}}
        self.stack = parser.Stack(self.ctx, 'update_test_stack', template.Template(tmpl, files={'foo': 'abc'}))
        self.stack.store()
        self.stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
        updated_stack = parser.Stack(self.ctx, 'updated_stack', template.Template(tmpl, files={'foo': 'xyz'}))

        def check_props_and_raise(*args):
            self.assertEqual('abc', self.stack['AResource'].properties['Foo'])
            raise resource.UpdateReplace()
        mock_update = self.patchobject(generic_rsrc.ResourceWithProps, 'update_template_diff', side_effect=check_props_and_raise)
        self.stack.update(updated_stack)
        self.assertEqual((parser.Stack.UPDATE, parser.Stack.COMPLETE), self.stack.state)
        self.assertEqual('xyz', self.stack['AResource'].properties['Foo'])
        mock_update.assert_called_once_with(rsrc_defn.ResourceDefinition('AResource', 'ResourceWithPropsType', properties={'Foo': 'xyz'}), rsrc_defn.ResourceDefinition('AResource', 'ResourceWithPropsType', properties={'Foo': 'abc'}))