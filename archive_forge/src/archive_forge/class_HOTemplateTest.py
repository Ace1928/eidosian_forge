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
class HOTemplateTest(common.HeatTestCase):
    """Test processing of HOT templates."""

    @staticmethod
    def resolve(snippet, template, stack=None):
        return function.resolve(template.parse(stack and stack.defn, snippet))

    @staticmethod
    def resolve_condition(snippet, template, stack=None):
        return function.resolve(template.parse_condition(stack and stack.defn, snippet))

    def test_defaults(self):
        """Test default content behavior of HOT template."""
        tmpl = template.Template(hot_tpl_empty)
        self.assertIsInstance(tmpl, hot_template.HOTemplate20130523)
        self.assertNotIn('foobar', tmpl)
        self.assertEqual('No description', tmpl[tmpl.DESCRIPTION])
        self.assertEqual({}, tmpl[tmpl.RESOURCES])
        self.assertEqual({}, tmpl[tmpl.OUTPUTS])

    def test_defaults_for_empty_sections(self):
        """Test default secntion's content behavior of HOT template."""
        tmpl = template.Template(hot_tpl_empty_sections)
        self.assertIsInstance(tmpl, hot_template.HOTemplate20130523)
        self.assertNotIn('foobar', tmpl)
        self.assertEqual('No description', tmpl[tmpl.DESCRIPTION])
        self.assertEqual({}, tmpl[tmpl.RESOURCES])
        self.assertEqual({}, tmpl[tmpl.OUTPUTS])
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        self.assertIsNone(stack.parameters._validate_user_parameters())
        self.assertIsNone(stack.validate())

    def test_translate_resources_good(self):
        """Test translation of resources into internal engine format."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          resource1:\n            type: AWS::EC2::Instance\n            properties:\n              property1: value1\n            metadata:\n              foo: bar\n            depends_on: dummy\n            deletion_policy: dummy\n            update_policy:\n              foo: bar\n        ')
        expected = {'resource1': {'Type': 'AWS::EC2::Instance', 'Properties': {'property1': 'value1'}, 'Metadata': {'foo': 'bar'}, 'DependsOn': 'dummy', 'DeletionPolicy': 'dummy', 'UpdatePolicy': {'foo': 'bar'}}}
        tmpl = template.Template(hot_tpl)
        self.assertEqual(expected, tmpl[tmpl.RESOURCES])

    def test_translate_resources_bad_no_data(self):
        """Test translation of resources without any mapping."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          resource1:\n        ')
        tmpl = template.Template(hot_tpl)
        error = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.RESOURCES)
        self.assertEqual('Each resource must contain a type key.', str(error))

    def test_translate_resources_bad_type(self):
        """Test translation of resources including invalid keyword."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          resource1:\n            Type: AWS::EC2::Instance\n            properties:\n              property1: value1\n            metadata:\n              foo: bar\n            depends_on: dummy\n            deletion_policy: dummy\n            update_policy:\n              foo: bar\n        ')
        tmpl = template.Template(hot_tpl)
        err = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.RESOURCES)
        self.assertEqual('"Type" is not a valid keyword inside a resource definition', str(err))

    def test_translate_resources_bad_properties(self):
        """Test translation of resources including invalid keyword."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          resource1:\n            type: AWS::EC2::Instance\n            Properties:\n              property1: value1\n            metadata:\n              foo: bar\n            depends_on: dummy\n            deletion_policy: dummy\n            update_policy:\n              foo: bar\n        ')
        tmpl = template.Template(hot_tpl)
        err = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.RESOURCES)
        self.assertEqual('"Properties" is not a valid keyword inside a resource definition', str(err))

    def test_translate_resources_resources_without_name(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          type: AWS::EC2::Instance\n          properties:\n            property1: value1\n          metadata:\n            foo: bar\n          depends_on: dummy\n          deletion_policy: dummy\n        ')
        tmpl = template.Template(hot_tpl)
        error = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.RESOURCES)
        self.assertEqual('"resources" must contain a map of resource maps. Found a [%s] instead' % str, str(error))

    def test_translate_resources_bad_metadata(self):
        """Test translation of resources including invalid keyword."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          resource1:\n            type: AWS::EC2::Instance\n            properties:\n              property1: value1\n            Metadata:\n              foo: bar\n            depends_on: dummy\n            deletion_policy: dummy\n            update_policy:\n              foo: bar\n        ')
        tmpl = template.Template(hot_tpl)
        err = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.RESOURCES)
        self.assertEqual('"Metadata" is not a valid keyword inside a resource definition', str(err))

    def test_translate_resources_bad_depends_on(self):
        """Test translation of resources including invalid keyword."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          resource1:\n            type: AWS::EC2::Instance\n            properties:\n              property1: value1\n            metadata:\n              foo: bar\n            DependsOn: dummy\n            deletion_policy: dummy\n            update_policy:\n              foo: bar\n        ')
        tmpl = template.Template(hot_tpl)
        err = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.RESOURCES)
        self.assertEqual('"DependsOn" is not a valid keyword inside a resource definition', str(err))

    def test_translate_resources_bad_deletion_policy(self):
        """Test translation of resources including invalid keyword."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          resource1:\n            type: AWS::EC2::Instance\n            properties:\n              property1: value1\n            metadata:\n              foo: bar\n            depends_on: dummy\n            DeletionPolicy: dummy\n            update_policy:\n              foo: bar\n        ')
        tmpl = template.Template(hot_tpl)
        err = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.RESOURCES)
        self.assertEqual('"DeletionPolicy" is not a valid keyword inside a resource definition', str(err))

    def test_translate_resources_bad_update_policy(self):
        """Test translation of resources including invalid keyword."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          resource1:\n            type: AWS::EC2::Instance\n            properties:\n              property1: value1\n            metadata:\n              foo: bar\n            depends_on: dummy\n            deletion_policy: dummy\n            UpdatePolicy:\n              foo: bar\n        ')
        tmpl = template.Template(hot_tpl)
        err = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.RESOURCES)
        self.assertEqual('"UpdatePolicy" is not a valid keyword inside a resource definition', str(err))

    def test_get_outputs_good(self):
        """Test get outputs."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        outputs:\n          output1:\n            description: output1\n            value: value1\n        ')
        expected = {'output1': {'description': 'output1', 'value': 'value1'}}
        tmpl = template.Template(hot_tpl)
        self.assertEqual(expected, tmpl[tmpl.OUTPUTS])

    def test_get_outputs_bad_no_data(self):
        """Test get outputs without any mapping."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        outputs:\n          output1:\n        ')
        tmpl = template.Template(hot_tpl)
        error = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.OUTPUTS)
        self.assertEqual('Each output must contain a value key.', str(error))

    def test_get_outputs_bad_without_name(self):
        """Test get outputs without name."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        outputs:\n          description: wrong output\n          value: value1\n        ')
        tmpl = template.Template(hot_tpl)
        error = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.OUTPUTS)
        self.assertEqual('"outputs" must contain a map of output maps. Found a [%s] instead' % str, str(error))

    def test_get_outputs_bad_description(self):
        """Test get outputs with bad description name."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        outputs:\n          output1:\n            Description: output1\n            value: value1\n        ')
        tmpl = template.Template(hot_tpl)
        err = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.OUTPUTS)
        self.assertIn('Description', str(err))

    def test_get_outputs_bad_value(self):
        """Test get outputs with bad value name."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        outputs:\n          output1:\n            description: output1\n            Value: value1\n        ')
        tmpl = template.Template(hot_tpl)
        err = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.OUTPUTS)
        self.assertIn('Value', str(err))

    def test_resource_group_list_join(self):
        """Test list_join on a ResourceGroup's inner attributes

        This should not fail during validation (i.e. before the ResourceGroup
        can return the list of the runtime values.
        """
        hot_tpl = template_format.parse('\n        heat_template_version: 2014-10-16\n        resources:\n          rg:\n            type: OS::Heat::ResourceGroup\n            properties:\n              count: 3\n              resource_def:\n                type: OS::Nova::Server\n        ')
        tmpl = template.Template(hot_tpl)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        snippet = {'list_join': ['\n', {'get_attr': ['rg', 'name']}]}
        self.assertEqual('', self.resolve(snippet, tmpl, stack))
        hot_tpl['heat_template_version'] = '2015-10-15'
        tmpl = template.Template(hot_tpl)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        snippet = {'list_join': ['\n', {'get_attr': ['rg', 'name']}]}
        self.assertEqual('', self.resolve(snippet, tmpl, stack))
        snippet = {'list_join': ['\n', {'get_attr': ['rg', 'name']}, {'get_attr': ['rg', 'name']}]}
        self.assertEqual('', self.resolve(snippet, tmpl, stack))

    def test_deletion_policy_titlecase(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2016-10-14\n        resources:\n          del:\n            type: OS::Heat::None\n            deletion_policy: Delete\n          ret:\n            type: OS::Heat::None\n            deletion_policy: Retain\n          snap:\n            type: OS::Heat::None\n            deletion_policy: Snapshot\n        ')
        rsrc_defns = template.Template(hot_tpl).resource_definitions(None)
        self.assertEqual(rsrc_defn.ResourceDefinition.DELETE, rsrc_defns['del'].deletion_policy())
        self.assertEqual(rsrc_defn.ResourceDefinition.RETAIN, rsrc_defns['ret'].deletion_policy())
        self.assertEqual(rsrc_defn.ResourceDefinition.SNAPSHOT, rsrc_defns['snap'].deletion_policy())

    def test_deletion_policy(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2016-10-14\n        resources:\n          del:\n            type: OS::Heat::None\n            deletion_policy: delete\n          ret:\n            type: OS::Heat::None\n            deletion_policy: retain\n          snap:\n            type: OS::Heat::None\n            deletion_policy: snapshot\n        ')
        rsrc_defns = template.Template(hot_tpl).resource_definitions(None)
        self.assertEqual(rsrc_defn.ResourceDefinition.DELETE, rsrc_defns['del'].deletion_policy())
        self.assertEqual(rsrc_defn.ResourceDefinition.RETAIN, rsrc_defns['ret'].deletion_policy())
        self.assertEqual(rsrc_defn.ResourceDefinition.SNAPSHOT, rsrc_defns['snap'].deletion_policy())

    def test_str_replace(self):
        """Test str_replace function."""
        snippet = {'str_replace': {'template': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar'}}}
        snippet_resolved = 'Template foo string bar'
        tmpl = template.Template(hot_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_str_replace_map_param(self):
        """Test old str_replace function with non-string map param."""
        snippet = {'str_replace': {'template': 'jsonvar1', 'params': {'jsonvar1': {'foo': 123}}}}
        tmpl = template.Template(hot_tpl_empty)
        ex = self.assertRaises(TypeError, self.resolve, snippet, tmpl)
        self.assertIn('"str_replace" params must be strings or numbers, param jsonvar1 is not valid', str(ex))

    def test_liberty_str_replace_map_param(self):
        """Test str_replace function with non-string map param."""
        snippet = {'str_replace': {'template': 'jsonvar1', 'params': {'jsonvar1': {'foo': 123}}}}
        snippet_resolved = '{"foo": 123}'
        tmpl = template.Template(hot_liberty_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_str_replace_list_param(self):
        """Test old str_replace function with non-string list param."""
        snippet = {'str_replace': {'template': 'listvar1', 'params': {'listvar1': ['foo', 123]}}}
        tmpl = template.Template(hot_tpl_empty)
        ex = self.assertRaises(TypeError, self.resolve, snippet, tmpl)
        self.assertIn('"str_replace" params must be strings or numbers, param listvar1 is not valid', str(ex))

    def test_liberty_str_replace_list_param(self):
        """Test str_replace function with non-string param."""
        snippet = {'str_replace': {'template': 'listvar1', 'params': {'listvar1': ['foo', 123]}}}
        snippet_resolved = '["foo", 123]'
        tmpl = template.Template(hot_liberty_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_str_replace_number(self):
        """Test str_replace function with numbers."""
        snippet = {'str_replace': {'template': 'Template number string bar', 'params': {'number': 1}}}
        snippet_resolved = 'Template 1 string bar'
        tmpl = template.Template(hot_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_str_fn_replace(self):
        """Test Fn:Replace function."""
        snippet = {'Fn::Replace': [{'$var1': 'foo', '$var2': 'bar'}, 'Template $var1 string $var2']}
        snippet_resolved = 'Template foo string bar'
        tmpl = template.Template(hot_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_str_replace_order(self):
        """Test str_replace function substitution order."""
        snippet = {'str_replace': {'template': '1234567890', 'params': {'1': 'a', '12': 'b', '123': 'c', '1234': 'd', '12345': 'e', '123456': 'f', '1234567': 'g'}}}
        tmpl = template.Template(hot_tpl_empty)
        self.assertEqual('g890', self.resolve(snippet, tmpl))

    def test_str_replace_single_pass(self):
        """Test that str_replace function does not do double substitution."""
        snippet = {'str_replace': {'template': '1234567890', 'params': {'1': 'a', '4': 'd', '8': 'h', '9': 'i', '123': '1', '456': '4', '890': '8', '90': '9'}}}
        tmpl = template.Template(hot_tpl_empty)
        self.assertEqual('1478', self.resolve(snippet, tmpl))

    def test_str_replace_sort_order(self):
        """Test str_replace function replacement order."""
        snippet = {'str_replace': {'template': '9892843210', 'params': {'989284': 'a', '892843': 'b', '765432': 'c', '654321': 'd', '543210': 'e'}}}
        tmpl = template.Template(hot_tpl_empty)
        self.assertEqual('9876e', self.resolve(snippet, tmpl))

    def test_str_replace_syntax(self):
        """Test str_replace function syntax.

        Pass wrong syntax (array instead of dictionary) to function and
        validate that we get a TypeError.
        """
        snippet = {'str_replace': [{'template': 'Template var1 string var2'}, {'params': {'var1': 'foo', 'var2': 'bar'}}]}
        tmpl = template.Template(hot_tpl_empty)
        self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)

    def test_str_replace_missing_param(self):
        """Test str_replace function missing param is OK."""
        snippet = {'str_replace': {'template': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar', 'var3': 'zed'}}}
        snippet_resolved = 'Template foo string bar'
        for hot_tpl in (hot_tpl_empty, hot_ocata_tpl_empty):
            tmpl = template.Template(hot_tpl)
            self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_str_replace_strict_no_missing_param(self):
        """Test str_replace_strict function no missing params, no problem."""
        snippet = {'str_replace_strict': {'template': 'Template var1 var1 s var2 t varvarvar3', 'params': {'var1': 'foo', 'var2': 'bar', 'var3': 'zed', 'var': 'tricky '}}}
        snippet_resolved = 'Template foo foo s bar t tricky tricky zed'
        tmpl = template.Template(hot_ocata_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_str_replace_strict_missing_param(self):
        """Test str_replace_strict function missing param(s) raises error."""
        snippet = {'str_replace_strict': {'template': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar', 'var3': 'zed'}}}
        tmpl = template.Template(hot_ocata_tpl_empty)
        ex = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
        self.assertEqual('The following params were not found in the template: var3', str(ex))
        snippet = {'str_replace_strict': {'template': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar', 'var0': 'zed'}}}
        ex = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
        self.assertEqual('The following params were not found in the template: var0', str(ex))
        snippet = {'str_replace_vstrict': {'template': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar', 'var0': 'zed', 'var': 'z', 'longvarname': 'q'}}}
        tmpl = template.Template(hot_pike_tpl_empty)
        ex = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
        self.assertEqual('The following params were not found in the template: longvarname,var0,var', str(ex))

    def test_str_replace_strict_empty_param_ok(self):
        """Test str_replace_strict function with empty params."""
        snippet = {'str_replace_strict': {'template': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': ''}}}
        tmpl = template.Template(hot_ocata_tpl_empty)
        self.assertEqual('Template foo string ', self.resolve(snippet, tmpl))

    def test_str_replace_vstrict_empty_param_not_ok(self):
        """Test str_replace_vstrict function with empty params.

        Raise ValueError when any of the params are None or empty.
        """
        snippet = {'str_replace_vstrict': {'template': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': ''}}}
        tmpl = template.Template(hot_pike_tpl_empty)
        for val in (None, '', {}, []):
            snippet['str_replace_vstrict']['params']['var2'] = val
            ex = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
            self.assertIn('str_replace_vstrict has an undefined or empty value for param var2', str(ex))

    def test_str_replace_invalid_param_keys(self):
        """Test str_replace function parameter keys.

        Pass wrong parameters to function and verify that we get
        a KeyError.
        """
        snippet = {'str_replace': {'tmpl': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar'}}}
        tmpl = template.Template(hot_tpl_empty)
        self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
        snippet = {'str_replace': {'tmpl': 'Template var1 string var2', 'parms': {'var1': 'foo', 'var2': 'bar'}}}
        ex = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
        self.assertIn('"str_replace" syntax should be str_replace:\\n', str(ex))

    def test_str_replace_strict_invalid_param_keys(self):
        """Test str_replace function parameter keys.

        Pass wrong parameters to function and verify that we get
        a KeyError.
        """
        snippets = [{'str_replace_strict': {'t': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar'}}}, {'str_replace_strict': {'template': 'Template var1 string var2', 'param': {'var1': 'foo', 'var2': 'bar'}}}]
        for snippet in snippets:
            tmpl = template.Template(hot_ocata_tpl_empty)
            ex = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
        self.assertIn('"str_replace_strict" syntax should be str_replace_strict:\\n', str(ex))

    def test_str_replace_invalid_param_types(self):
        """Test str_replace function parameter values.

        Pass parameter values of wrong type to function and verify that we get
        a TypeError.
        """
        snippet = {'str_replace': {'template': 12345, 'params': {'var1': 'foo', 'var2': 'bar'}}}
        tmpl = template.Template(hot_tpl_empty)
        self.assertRaises(TypeError, self.resolve, snippet, tmpl)
        snippet = {'str_replace': {'template': 'Template var1 string var2', 'params': ['var1', 'foo', 'var2', 'bar']}}
        ex = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
        self.assertIn('str_replace: "str_replace" parameters must be a mapping', str(ex))

    def test_str_replace_invalid_param_type_init(self):
        """Test str_replace function parameter values.

        Pass parameter values of wrong type to function and verify that we get
        a TypeError in the constructor.
        """
        args = [['var1', 'foo', 'var2', 'bar'], 'Template var1 string var2']
        ex = self.assertRaises(TypeError, cfn_functions.Replace, None, 'Fn::Replace', args)
        self.assertIn('parameters must be a mapping', str(ex))

    def test_str_replace_ref_get_param(self):
        """Test str_replace referencing parameters."""
        hot_tpl = template_format.parse('\n        heat_template_version: 2015-04-30\n        parameters:\n          p_template:\n            type: string\n            default: foo-replaceme\n          p_params:\n            type: json\n            default:\n              replaceme: success\n        resources:\n          rsrc:\n            type: ResWithStringPropAndAttr\n            properties:\n              a_string:\n                str_replace:\n                  template: {get_param: p_template}\n                  params: {get_param: p_params}\n        outputs:\n          replaced:\n            value: {get_attr: [rsrc, string]}\n        ')
        tmpl = template.Template(hot_tpl)
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        self.stack.store()
        self.stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
        self.stack._update_all_resource_data(False, True)
        self.assertEqual('foo-success', self.stack.outputs['replaced'].get_value())

    def test_get_file(self):
        """Test get_file function."""
        snippet = {'get_file': 'file:///tmp/foo.yaml'}
        snippet_resolved = 'foo contents'
        tmpl = template.Template(hot_tpl_empty, files={'file:///tmp/foo.yaml': 'foo contents'})
        stack = parser.Stack(utils.dummy_context(), 'param_id_test', tmpl)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl, stack))

    def test_get_file_not_string(self):
        """Test get_file function with non-string argument."""
        snippet = {'get_file': ['file:///tmp/foo.yaml']}
        tmpl = template.Template(hot_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'param_id_test', tmpl)
        notStrErr = self.assertRaises(TypeError, self.resolve, snippet, tmpl, stack)
        self.assertEqual('Argument to "get_file" must be a string', str(notStrErr))

    def test_get_file_missing_files(self):
        """Test get_file function with no matching key in files section."""
        snippet = {'get_file': 'file:///tmp/foo.yaml'}
        tmpl = template.Template(hot_tpl_empty, files={'file:///tmp/bar.yaml': 'bar contents'})
        stack = parser.Stack(utils.dummy_context(), 'param_id_test', tmpl)
        missingErr = self.assertRaises(ValueError, self.resolve, snippet, tmpl, stack)
        self.assertEqual('No content found in the "files" section for get_file path: file:///tmp/foo.yaml', str(missingErr))

    def test_get_file_nested_does_not_resolve(self):
        """Test get_file function does not resolve nested calls."""
        snippet = {'get_file': 'file:///tmp/foo.yaml'}
        snippet_resolved = '{get_file: file:///tmp/bar.yaml}'
        tmpl = template.Template(hot_tpl_empty, files={'file:///tmp/foo.yaml': snippet_resolved, 'file:///tmp/bar.yaml': 'bar content'})
        stack = parser.Stack(utils.dummy_context(), 'param_id_test', tmpl)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl, stack))

    def test_list_join(self):
        snippet = {'list_join': [',', ['bar', 'baz']]}
        snippet_resolved = 'bar,baz'
        tmpl = template.Template(hot_kilo_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_join_multiple(self):
        snippet = {'list_join': [',', ['bar', 'baz'], ['bar2', 'baz2']]}
        snippet_resolved = 'bar,baz,bar2,baz2'
        tmpl = template.Template(hot_liberty_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_list_join_empty_list(self):
        snippet = {'list_join': [',', []]}
        snippet_resolved = ''
        k_tmpl = template.Template(hot_kilo_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, k_tmpl))
        l_tmpl = template.Template(hot_liberty_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, l_tmpl))

    def test_join_json(self):
        snippet = {'list_join': [',', [{'foo': 'json'}, {'foo2': 'json2'}]]}
        snippet_resolved = '{"foo": "json"},{"foo2": "json2"}'
        l_tmpl = template.Template(hot_liberty_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, l_tmpl))
        k_tmpl = template.Template(hot_kilo_tpl_empty)
        exc = self.assertRaises(TypeError, self.resolve, snippet, k_tmpl)
        self.assertEqual("Items to join must be strings not {'foo': 'json'}", str(exc))

    def test_join_object_type_fail(self):
        not_serializable = object
        snippet = {'list_join': [',', [not_serializable]]}
        l_tmpl = template.Template(hot_liberty_tpl_empty)
        exc = self.assertRaises(TypeError, self.resolve, snippet, l_tmpl)
        self.assertIn('Items to join must be string, map or list not', str(exc))
        k_tmpl = template.Template(hot_kilo_tpl_empty)
        exc = self.assertRaises(TypeError, self.resolve, snippet, k_tmpl)
        self.assertIn('Items to join must be strings', str(exc))

    def test_join_json_fail(self):
        not_serializable = object
        snippet = {'list_join': [',', [{'foo': not_serializable}]]}
        l_tmpl = template.Template(hot_liberty_tpl_empty)
        exc = self.assertRaises(TypeError, self.resolve, snippet, l_tmpl)
        self.assertIn('Items to join must be string, map or list', str(exc))
        self.assertIn('failed json serialization', str(exc))

    def test_join_invalid(self):
        snippet = {'list_join': 'bad'}
        l_tmpl = template.Template(hot_liberty_tpl_empty)
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, l_tmpl)
        self.assertIn('list_join: Incorrect arguments to "list_join"', str(exc))
        k_tmpl = template.Template(hot_kilo_tpl_empty)
        exc1 = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, k_tmpl)
        self.assertIn('list_join: Incorrect arguments to "list_join"', str(exc1))

    def test_join_int_invalid(self):
        snippet = {'list_join': 5}
        l_tmpl = template.Template(hot_liberty_tpl_empty)
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, l_tmpl)
        self.assertIn('list_join: Incorrect arguments', str(exc))
        k_tmpl = template.Template(hot_kilo_tpl_empty)
        exc1 = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, k_tmpl)
        self.assertIn('list_join: Incorrect arguments', str(exc1))

    def test_join_invalid_value(self):
        snippet = {'list_join': [',']}
        l_tmpl = template.Template(hot_liberty_tpl_empty)
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, l_tmpl)
        self.assertIn('list_join: Incorrect arguments to "list_join"', str(exc))
        k_tmpl = template.Template(hot_kilo_tpl_empty)
        exc1 = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, k_tmpl)
        self.assertIn('list_join: Incorrect arguments to "list_join"', str(exc1))

    def test_join_invalid_multiple(self):
        snippet = {'list_join': [',', 'bad', ['foo']]}
        tmpl = template.Template(hot_liberty_tpl_empty)
        exc = self.assertRaises(TypeError, self.resolve, snippet, tmpl)
        self.assertIn('must operate on a list', str(exc))

    def test_merge(self):
        snippet = {'map_merge': [{'f1': 'b1', 'f2': 'b2'}, {'f1': 'b2'}]}
        tmpl = template.Template(hot_mitaka_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('b2', resolved['f1'])
        self.assertEqual('b2', resolved['f2'])

    def test_merge_none(self):
        snippet = {'map_merge': [{'f1': 'b1', 'f2': 'b2'}, None]}
        tmpl = template.Template(hot_mitaka_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('b1', resolved['f1'])
        self.assertEqual('b2', resolved['f2'])

    def test_merge_invalid(self):
        snippet = {'map_merge': [{'f1': 'b1', 'f2': 'b2'}, ['f1', 'b2']]}
        tmpl = template.Template(hot_mitaka_tpl_empty)
        exc = self.assertRaises(TypeError, self.resolve, snippet, tmpl)
        self.assertIn('Incorrect arguments', str(exc))

    def test_merge_containing_repeat(self):
        snippet = {'map_merge': {'repeat': {'template': {'ROLE': 'ROLE'}, 'for_each': {'ROLE': ['role1', 'role2']}}}}
        tmpl = template.Template(hot_mitaka_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('role1', resolved['role1'])
        self.assertEqual('role2', resolved['role2'])

    def test_merge_containing_repeat_with_none(self):
        snippet = {'map_merge': {'repeat': {'template': {'ROLE': 'ROLE'}, 'for_each': {'ROLE': None}}}}
        tmpl = template.Template(hot_mitaka_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual({}, resolved)

    def test_merge_containing_repeat_multi_list_no_nested_loop_with_none(self):
        snippet = {'map_merge': {'repeat': {'template': {'ROLE': 'ROLE', 'NAME': 'NAME'}, 'for_each': {'ROLE': None, 'NAME': ['n1', 'n2']}, 'permutations': False}}}
        tmpl = template.Template(hot_mitaka_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual({}, resolved)

    def test_merge_containing_repeat_multi_list_no_nested_loop_all_none(self):
        snippet = {'map_merge': {'repeat': {'template': {'ROLE': 'ROLE', 'NAME': 'NAME'}, 'for_each': {'ROLE': None, 'NAME': None}, 'permutations': False}}}
        tmpl = template.Template(hot_mitaka_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual({}, resolved)

    def test_map_replace(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': 'b2'}, {'keys': {'f1': 'F1'}, 'values': {'b2': 'B2'}}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual({'F1': 'b1', 'f2': 'B2'}, resolved)

    def test_map_replace_nokeys(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': 'b2'}, {'values': {'b2': 'B2'}}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual({'f1': 'b1', 'f2': 'B2'}, resolved)

    def test_map_replace_novalues(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': 'b2'}, {'keys': {'f2': 'F2'}}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual({'f1': 'b1', 'F2': 'b2'}, resolved)

    def test_map_replace_keys_collide_ok_equal(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': 'b2'}, {'keys': {'f2': 'f2'}}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual({'f1': 'b1', 'f2': 'b2'}, resolved)

    def test_map_replace_none_values(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': 'b2'}, {'values': None}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual({'f1': 'b1', 'f2': 'b2'}, resolved)

    def test_map_replace_none_keys(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': 'b2'}, {'keys': None}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual({'f1': 'b1', 'f2': 'b2'}, resolved)

    def test_map_replace_unhashable_value(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': []}, {'values': {}}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual({'f1': 'b1', 'f2': []}, resolved)

    def test_map_replace_keys_collide(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': 'b2'}, {'keys': {'f2': 'f1'}}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        msg = 'key replacement f1 collides with a key in the input map'
        self.assertRaisesRegex(ValueError, msg, self.resolve, snippet, tmpl)

    def test_map_replace_replaced_keys_collide(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': 'b2'}, {'keys': {'f1': 'f3', 'f2': 'f3'}}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        msg = 'key replacement f3 collides with a key in the output map'
        self.assertRaisesRegex(ValueError, msg, self.resolve, snippet, tmpl)

    def test_map_replace_invalid_str_arg1(self):
        snippet = {'map_replace': 'ab'}
        tmpl = template.Template(hot_newton_tpl_empty)
        msg = 'Incorrect arguments to "map_replace" should be:'
        self.assertRaisesRegex(TypeError, msg, self.resolve, snippet, tmpl)

    def test_map_replace_invalid_str_arg2(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': 'b2'}, 'ab']}
        tmpl = template.Template(hot_newton_tpl_empty)
        msg = 'Incorrect arguments: to "map_replace", arguments must be a list of maps'
        self.assertRaisesRegex(TypeError, msg, self.resolve, snippet, tmpl)

    def test_map_replace_invalid_empty(self):
        snippet = {'map_replace': []}
        tmpl = template.Template(hot_newton_tpl_empty)
        msg = 'Incorrect arguments to "map_replace" should be:'
        self.assertRaisesRegex(TypeError, msg, self.resolve, snippet, tmpl)

    def test_map_replace_invalid_missing1(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': 'b2'}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        msg = 'Incorrect arguments to "map_replace" should be:'
        self.assertRaisesRegex(TypeError, msg, self.resolve, snippet, tmpl)

    def test_map_replace_invalid_missing2(self):
        snippet = {'map_replace': [{'keys': {'f1': 'f3', 'f2': 'f3'}}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        msg = 'Incorrect arguments to "map_replace" should be:'
        self.assertRaisesRegex(TypeError, msg, self.resolve, snippet, tmpl)

    def test_map_replace_invalid_wrongkey(self):
        snippet = {'map_replace': [{'f1': 'b1', 'f2': 'b2'}, {'notkeys': {'f2': 'F2'}}]}
        tmpl = template.Template(hot_newton_tpl_empty)
        msg = 'Incorrect arguments to "map_replace" should be:'
        self.assertRaisesRegex(ValueError, msg, self.resolve, snippet, tmpl)

    def test_yaql(self):
        snippet = {'yaql': {'expression': '$.data.var1.sum()', 'data': {'var1': [1, 2, 3, 4]}}}
        tmpl = template.Template(hot_newton_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        resolved = self.resolve(snippet, tmpl, stack=stack)
        self.assertEqual(10, resolved)

    def test_yaql_list_input(self):
        snippet = {'yaql': {'expression': '$.data.sum()', 'data': [1, 2, 3, 4]}}
        tmpl = template.Template(hot_newton_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        resolved = self.resolve(snippet, tmpl, stack=stack)
        self.assertEqual(10, resolved)

    def test_yaql_string_input(self):
        snippet = {'yaql': {'expression': '$.data', 'data': 'whynotastring'}}
        tmpl = template.Template(hot_newton_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        resolved = self.resolve(snippet, tmpl, stack=stack)
        self.assertEqual('whynotastring', resolved)

    def test_yaql_int_input(self):
        snippet = {'yaql': {'expression': '$.data + 2', 'data': 2}}
        tmpl = template.Template(hot_newton_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        resolved = self.resolve(snippet, tmpl, stack=stack)
        self.assertEqual(4, resolved)

    def test_yaql_bogus_keys(self):
        snippet = {'yaql': {'expression': '1 + 3', 'data': {'var1': [1, 2, 3, 4]}, 'bogus': ''}}
        tmpl = template.Template(hot_newton_tpl_empty)
        self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)

    def test_yaql_invalid_syntax(self):
        snippet = {'yaql': {'wrong': 'wrong_expr', 'wrong_data': 'mustbeamap'}}
        tmpl = template.Template(hot_newton_tpl_empty)
        self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)

    def test_yaql_non_map_args(self):
        snippet = {'yaql': 'invalid'}
        tmpl = template.Template(hot_newton_tpl_empty)
        msg = 'yaql: Arguments to "yaql" must be a map.'
        self.assertRaisesRegex(exception.StackValidationFailed, msg, self.resolve, snippet, tmpl)

    def test_yaql_invalid_expression(self):
        snippet = {'yaql': {'expression': 'invalid(', 'data': {'var1': [1, 2, 3, 4]}}}
        tmpl = template.Template(hot_newton_tpl_empty)
        yaql = tmpl.parse(None, snippet)
        regxp = 'yaql: Bad expression Parse error: unexpected end of statement.'
        self.assertRaisesRegex(exception.StackValidationFailed, regxp, function.validate, yaql)

    def test_yaql_data_as_function(self):
        snippet = {'yaql': {'expression': '$.data.var1.len()', 'data': {'var1': {'list_join': ['', ['1', '2']]}}}}
        tmpl = template.Template(hot_newton_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        resolved = self.resolve(snippet, tmpl, stack=stack)
        self.assertEqual(2, resolved)

    def test_yaql_merge(self):
        snippet = {'yaql': {'expression': '$.data.d.reduce($1.mergeWith($2))', 'data': {'d': [{'a': [1]}, {'a': [2]}, {'a': [3]}]}}}
        tmpl = template.Template(hot_newton_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        resolved = self.resolve(snippet, tmpl, stack=stack)
        self.assertEqual({'a': [1, 2, 3]}, resolved)

    def test_yaql_as_condition(self):
        hot_tpl = template_format.parse("\n        heat_template_version: pike\n        parameters:\n          ServiceNames:\n            type: comma_delimited_list\n            default: ['neutron', 'heat']\n        ")
        snippet = {'yaql': {'expression': '$.data.service_names.contains("neutron")', 'data': {'service_names': {'get_param': 'ServiceNames'}}}}
        tmpl = template.Template(hot_tpl)
        stack = parser.Stack(utils.dummy_context(), 'test_condition_yaql_true', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stack)
        self.assertTrue(resolved)
        tmpl = template.Template(hot_tpl, env=environment.Environment({'ServiceNames': ['nova_network', 'heat']}))
        stack = parser.Stack(utils.dummy_context(), 'test_condition_yaql_false', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stack)
        self.assertFalse(resolved)

    def test_equals(self):
        hot_tpl = template_format.parse("\n        heat_template_version: 2016-10-14\n        parameters:\n          env_type:\n            type: string\n            default: 'test'\n        ")
        snippet = {'equals': [{'get_param': 'env_type'}, 'prod']}
        tmpl = template.Template(hot_tpl)
        stack = parser.Stack(utils.dummy_context(), 'test_equals_false', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stack)
        self.assertFalse(resolved)
        tmpl = template.Template(hot_tpl, env=environment.Environment({'env_type': 'prod'}))
        stack = parser.Stack(utils.dummy_context(), 'test_equals_true', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stack)
        self.assertTrue(resolved)

    def test_equals_invalid_args(self):
        tmpl = template.Template(hot_newton_tpl_empty)
        snippet = {'equals': ['test', 'prod', 'invalid']}
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
        error_msg = 'equals: Arguments to "equals" must be of the form: [value_1, value_2]'
        self.assertIn(error_msg, str(exc))
        snippet = {'equals': 'invalid condition'}
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
        self.assertIn(error_msg, str(exc))

    def test_equals_with_non_supported_function(self):
        tmpl = template.Template(hot_newton_tpl_empty)
        snippet = {'equals': [{'get_attr': [None, 'att1']}, {'get_attr': [None, 'att2']}]}
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
        self.assertIn('"get_attr" is invalid', str(exc))

    def test_if(self):
        snippet = {'if': ['create_prod', 'value_if_true', 'value_if_false']}
        tmpl = template.Template(hot_newton_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_if_function', tmpl)
        with mock.patch.object(tmpl, 'conditions') as conds:
            conds.return_value = conditions.Conditions({'create_prod': True})
            resolved = self.resolve(snippet, tmpl, stack)
            self.assertEqual('value_if_true', resolved)
        with mock.patch.object(tmpl, 'conditions') as conds:
            conds.return_value = conditions.Conditions({'create_prod': False})
            resolved = self.resolve(snippet, tmpl, stack)
            self.assertEqual('value_if_false', resolved)

    def test_if_using_boolean_condition(self):
        snippet = {'if': [True, 'value_if_true', 'value_if_false']}
        tmpl = template.Template(hot_newton_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_if_using_boolean_condition', tmpl)
        resolved = self.resolve(snippet, tmpl, stack)
        self.assertEqual('value_if_true', resolved)
        snippet = {'if': [False, 'value_if_true', 'value_if_false']}
        resolved = self.resolve(snippet, tmpl, stack)
        self.assertEqual('value_if_false', resolved)

    def test_if_null_return(self):
        snippet = {'if': [True, None, 'value_if_false']}
        tmpl = template.Template(hot_newton_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_if_null_return', tmpl)
        resolved = self.resolve(snippet, tmpl, stack)
        self.assertIsNone(resolved)

    def test_if_using_condition_function(self):
        tmpl_with_conditions = template_format.parse('\nheat_template_version: 2016-10-14\nconditions:\n  create_prod: False\n')
        snippet = {'if': [{'not': 'create_prod'}, 'value_if_true', 'value_if_false']}
        tmpl = template.Template(tmpl_with_conditions)
        stack = parser.Stack(utils.dummy_context(), 'test_if_using_condition_function', tmpl)
        resolved = self.resolve(snippet, tmpl, stack)
        self.assertEqual('value_if_true', resolved)

    def test_if_referenced_by_resource(self):
        tmpl_with_conditions = template_format.parse('\nheat_template_version: pike\nconditions:\n  create_prod: False\nresources:\n  AResource:\n    type: ResourceWithPropsType\n    properties:\n      Foo:\n        if:\n          - create_prod\n          - "one"\n          - "two"\n')
        tmpl = template.Template(tmpl_with_conditions)
        self.stack = parser.Stack(utils.dummy_context(), 'test_if_referenced_by_resource', tmpl)
        self.stack.store()
        self.stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
        self.assertEqual('two', self.stack['AResource'].properties['Foo'])

    def test_if_referenced_by_resource_null(self):
        tmpl_with_conditions = template_format.parse('\nheat_template_version: pike\nconditions:\n  create_prod: True\nresources:\n  AResource:\n    type: ResourceWithPropsType\n    properties:\n      Foo:\n        if:\n          - create_prod\n          - null\n          - "two"\n')
        tmpl = template.Template(tmpl_with_conditions)
        self.stack = parser.Stack(utils.dummy_context(), 'test_if_referenced_by_resource_null', tmpl)
        self.stack.store()
        self.stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
        self.assertEqual('', self.stack['AResource'].properties['Foo'])

    def test_if_invalid_args(self):
        snippets = [{'if': ['create_prod', 'one_value']}, {'if': ['create_prod', 'one_value', 'two_values', 'three_values']}]
        tmpl = template.Template(hot_newton_tpl_empty)
        for snippet in snippets:
            exc = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
            self.assertIn('Arguments to "if" must be of the form: [condition_name, value_if_true, value_if_false]', str(exc))

    def test_if_nullable_invalid_args(self):
        snippets = [{'if': ['create_prod']}, {'if': ['create_prod', 'one_value', 'two_values', 'three_values']}]
        tmpl = template.Template(hot_wallaby_tpl_empty)
        for snippet in snippets:
            exc = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
            self.assertIn('Arguments to "if" must be of the form: [condition_name, value_if_true, value_if_false]', str(exc))

    def test_if_nullable(self):
        snippet = {'single': {'if': [False, 'value_if_true']}, 'nested_true': {'if': [True, {'if': [False, 'foo']}, 'bar']}, 'nested_false': {'if': [False, 'baz', {'if': [False, 'quux']}]}, 'control': {'if': [False, True, None]}}
        tmpl = template.Template(hot_wallaby_tpl_empty)
        resolved = self.resolve(snippet, tmpl, None)
        self.assertEqual({'control': None}, resolved)

    def test_if_condition_name_non_existing(self):
        snippet = {'if': ['cd_not_existing', 'value_true', 'value_false']}
        tmpl = template.Template(hot_newton_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_if_function', tmpl)
        with mock.patch.object(tmpl, 'conditions') as conds:
            conds.return_value = conditions.Conditions({'create_prod': True})
            exc = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl, stack)
        self.assertIn('Invalid condition "cd_not_existing"', str(exc))
        self.assertIn('if:', str(exc))

    def _test_repeat(self, templ=hot_kilo_tpl_empty):
        """Test repeat function."""
        snippet = {'repeat': {'template': 'this is %var%', 'for_each': {'%var%': ['a', 'b', 'c']}}}
        snippet_resolved = ['this is a', 'this is b', 'this is c']
        tmpl = template.Template(templ)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_repeat(self):
        self._test_repeat()

    def test_repeat_with_pike_version(self):
        self._test_repeat(templ=hot_pike_tpl_empty)

    def test_repeat_get_param(self):
        """Test repeat function with get_param function as an argument."""
        hot_tpl = template_format.parse("\n        heat_template_version: 2015-04-30\n        parameters:\n          param:\n            type: comma_delimited_list\n            default: 'a,b,c'\n        ")
        snippet = {'repeat': {'template': 'this is var%', 'for_each': {'var%': {'get_param': 'param'}}}}
        snippet_resolved = ['this is a', 'this is b', 'this is c']
        tmpl = template.Template(hot_tpl)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl, stack))

    def _test_repeat_dict_with_no_replacement(self, templ=hot_newton_tpl_empty):
        snippet = {'repeat': {'template': {'SERVICE_enabled': True}, 'for_each': {'SERVICE': ['x', 'y', 'z']}}}
        snippet_resolved = [{'x_enabled': True}, {'y_enabled': True}, {'z_enabled': True}]
        tmpl = template.Template(templ)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_repeat_dict_with_no_replacement(self):
        self._test_repeat_dict_with_no_replacement()

    def test_repeat_dict_with_no_replacement_pike_version(self):
        self._test_repeat_dict_with_no_replacement(templ=hot_pike_tpl_empty)

    def _test_repeat_dict_template(self, templ=hot_kilo_tpl_empty):
        """Test repeat function with a dictionary as a template."""
        snippet = {'repeat': {'template': {'key-%var%': 'this is %var%'}, 'for_each': {'%var%': ['a', 'b', 'c']}}}
        snippet_resolved = [{'key-a': 'this is a'}, {'key-b': 'this is b'}, {'key-c': 'this is c'}]
        tmpl = template.Template(templ)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_repeat_dict_template(self):
        self._test_repeat_dict_template()

    def test_repeat_dict_template_pike_version(self):
        self._test_repeat_dict_template(templ=hot_pike_tpl_empty)

    def _test_repeat_list_template(self, templ=hot_kilo_tpl_empty):
        """Test repeat function with a list as a template."""
        snippet = {'repeat': {'template': ['this is %var%', 'static'], 'for_each': {'%var%': ['a', 'b', 'c']}}}
        snippet_resolved = [['this is a', 'static'], ['this is b', 'static'], ['this is c', 'static']]
        tmpl = template.Template(templ)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_repeat_list_template(self):
        self._test_repeat_list_template()

    def test_repeat_list_template_pike_version(self):
        self._test_repeat_list_template(templ=hot_pike_tpl_empty)

    def _test_repeat_multi_list(self, templ=hot_kilo_tpl_empty):
        """Test repeat function with multiple input lists."""
        snippet = {'repeat': {'template': 'this is %var1%-%var2%', 'for_each': {'%var1%': ['a', 'b', 'c'], '%var2%': ['1', '2']}}}
        snippet_resolved = ['this is a-1', 'this is b-1', 'this is c-1', 'this is a-2', 'this is b-2', 'this is c-2']
        tmpl = template.Template(templ)
        result = self.resolve(snippet, tmpl)
        self.assertEqual(len(result), len(snippet_resolved))
        for item in result:
            self.assertIn(item, snippet_resolved)

    def test_repeat_multi_list(self):
        self._test_repeat_multi_list()

    def test_repeat_multi_list_pike_version(self):
        self._test_repeat_multi_list(templ=hot_pike_tpl_empty)

    def test_repeat_list_and_map(self):
        """Test repeat function with a list and a map."""
        snippet = {'repeat': {'template': 'this is %var1%-%var2%', 'for_each': {'%var1%': ['a', 'b', 'c'], '%var2%': {'x': 'v', 'y': 'v'}}}}
        snippet_resolved = ['this is a-x', 'this is b-x', 'this is c-x', 'this is a-y', 'this is b-y', 'this is c-y']
        tmpl = template.Template(hot_newton_tpl_empty)
        result = self.resolve(snippet, tmpl)
        self.assertEqual(len(result), len(snippet_resolved))
        for item in result:
            self.assertIn(item, snippet_resolved)

    def test_repeat_with_no_nested_loop(self):
        snippet = {'repeat': {'template': {'network': '%net%', 'port': '%port%', 'subnet': '%sub%'}, 'for_each': {'%net%': ['n1', 'n2', 'n3', 'n4'], '%port%': ['p1', 'p2', 'p3', 'p4'], '%sub%': ['s1', 's2', 's3', 's4']}, 'permutations': False}}
        tmpl = template.Template(hot_pike_tpl_empty)
        snippet_resolved = [{'network': 'n1', 'port': 'p1', 'subnet': 's1'}, {'network': 'n2', 'port': 'p2', 'subnet': 's2'}, {'network': 'n3', 'port': 'p3', 'subnet': 's3'}, {'network': 'n4', 'port': 'p4', 'subnet': 's4'}]
        result = self.resolve(snippet, tmpl)
        self.assertEqual(snippet_resolved, result)

    def test_repeat_no_nested_loop_different_len(self):
        snippet = {'repeat': {'template': {'network': '%net%', 'port': '%port%', 'subnet': '%sub%'}, 'for_each': {'%net%': ['n1', 'n2', 'n3'], '%port%': ['p1', 'p2'], '%sub%': ['s1', 's2']}, 'permutations': False}}
        tmpl = template.Template(hot_pike_tpl_empty)
        self.assertRaises(ValueError, self.resolve, snippet, tmpl)

    def test_repeat_no_nested_loop_with_dict_type(self):
        snippet = {'repeat': {'template': {'network': '%net%', 'port': '%port%', 'subnet': '%sub%'}, 'for_each': {'%net%': ['n1', 'n2'], '%port%': {'p1': 'pp', 'p2': 'qq'}, '%sub%': ['s1', 's2']}, 'permutations': False}}
        tmpl = template.Template(hot_pike_tpl_empty)
        self.assertRaises(TypeError, self.resolve, snippet, tmpl)

    def test_repeat_permutations_non_bool(self):
        snippet = {'repeat': {'template': {'network': '%net%', 'port': '%port%', 'subnet': '%sub%'}, 'for_each': {'%net%': ['n1', 'n2'], '%port%': ['p1', 'p2'], '%sub%': ['s1', 's2']}, 'permutations': 'non bool'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
        self.assertIn('"permutations" should be boolean type for repeat function', str(exc))

    def test_repeat_bad_args(self):
        """Tests reporting error by repeat function.

        Test that the repeat function reports a proper error when missing or
        invalid arguments.
        """
        tmpl = template.Template(hot_kilo_tpl_empty)
        snippet = {'repeat': {'template': 'this is %var%'}}
        self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
        snippet = {'repeat': {'template': 'this is %var%', 'foreach': {'%var%': ['a', 'b', 'c']}}}
        self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
        snippet = {'repeat': {'templte': 'this is %var%', 'for_each': {'%var%': ['a', 'b', 'c']}}}
        self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)

    def test_repeat_bad_arg_type(self):
        tmpl = template.Template(hot_kilo_tpl_empty)
        snippet = {'repeat': {'template': 'this is %var%', 'for_each': '%var%'}}
        repeat = tmpl.parse(None, snippet)
        regxp = 'repeat: The "for_each" argument to "repeat" must contain a map'
        self.assertRaisesRegex(exception.StackValidationFailed, regxp, function.validate, repeat)

    def test_digest(self):
        snippet = {'digest': ['md5', 'foobar']}
        snippet_resolved = '3858f62230ac3c915f300c664312c63f'
        tmpl = template.Template(hot_kilo_tpl_empty)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_digest_invalid_types(self):
        tmpl = template.Template(hot_kilo_tpl_empty)
        invalid_snippets = [{'digest': 'invalid'}, {'digest': {'foo': 'invalid'}}, {'digest': [123]}]
        for snippet in invalid_snippets:
            exc = self.assertRaises(TypeError, self.resolve, snippet, tmpl)
            self.assertIn('must be a list of strings', str(exc))

    def test_digest_incorrect_number_arguments(self):
        tmpl = template.Template(hot_kilo_tpl_empty)
        invalid_snippets = [{'digest': []}, {'digest': ['foo']}, {'digest': ['md5']}, {'digest': ['md5', 'foo', 'bar']}]
        for snippet in invalid_snippets:
            exc = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
            self.assertIn('usage: ["<algorithm>", "<value>"]', str(exc))

    def test_digest_invalid_algorithm(self):
        tmpl = template.Template(hot_kilo_tpl_empty)
        snippet = {'digest': ['invalid_algorithm', 'foobar']}
        exc = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
        self.assertIn('Algorithm must be one of', str(exc))

    def test_str_split(self):
        tmpl = template.Template(hot_liberty_tpl_empty)
        snippet = {'str_split': [',', 'bar,baz']}
        snippet_resolved = ['bar', 'baz']
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_str_split_index(self):
        tmpl = template.Template(hot_liberty_tpl_empty)
        snippet = {'str_split': [',', 'bar,baz', 1]}
        snippet_resolved = 'baz'
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_str_split_index_str(self):
        tmpl = template.Template(hot_liberty_tpl_empty)
        snippet = {'str_split': [',', 'bar,baz', '1']}
        snippet_resolved = 'baz'
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))

    def test_str_split_index_bad(self):
        tmpl = template.Template(hot_liberty_tpl_empty)
        snippet = {'str_split': [',', 'bar,baz', 'bad']}
        exc = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
        self.assertIn('Incorrect index to "str_split"', str(exc))

    def test_str_split_index_out_of_range(self):
        tmpl = template.Template(hot_liberty_tpl_empty)
        snippet = {'str_split': [',', 'bar,baz', '2']}
        exc = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
        expected = 'Incorrect index to "str_split" should be between 0 and 1'
        self.assertEqual(expected, str(exc))

    def test_str_split_bad_novalue(self):
        tmpl = template.Template(hot_liberty_tpl_empty)
        snippet = {'str_split': [',']}
        exc = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
        self.assertIn('Incorrect arguments to "str_split"', str(exc))

    def test_str_split_bad_empty(self):
        tmpl = template.Template(hot_liberty_tpl_empty)
        snippet = {'str_split': []}
        exc = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
        self.assertIn('Incorrect arguments to "str_split"', str(exc))

    def test_str_split_none_string_to_split(self):
        tmpl = template.Template(hot_liberty_tpl_empty)
        snippet = {'str_split': ['.', None]}
        self.assertIsNone(self.resolve(snippet, tmpl))

    def test_str_split_none_delim(self):
        tmpl = template.Template(hot_liberty_tpl_empty)
        snippet = {'str_split': [None, 'check']}
        self.assertEqual(['check'], self.resolve(snippet, tmpl))

    def test_prevent_parameters_access(self):
        """Check parameters section inaccessible using the template as a dict.

        Test that the parameters section can't be accessed using the template
        as a dictionary.
        """
        expected_description = 'This can be accessed'
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        description: {0}\n        parameters:\n          foo:\n            type: string\n        '.format(expected_description))
        tmpl = template.Template(hot_tpl)
        self.assertEqual(expected_description, tmpl['description'])
        err_str = 'can not be accessed directly'
        keyError = self.assertRaises(KeyError, tmpl.__getitem__, 'parameters')
        self.assertIn(err_str, str(keyError))
        keyError = self.assertRaises(KeyError, tmpl.__getitem__, 'Parameters')
        self.assertIn(err_str, str(keyError))

    def test_parameters_section_not_iterable(self):
        """Check parameters section is not returned using the template as iter.

        Test that the parameters section is not returned when the template is
        used as an iterable.
        """
        expected_description = 'This can be accessed'
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'description': expected_description, 'parameters': {'foo': {'Type': 'String', 'Required': True}}})
        self.assertEqual(expected_description, tmpl['description'])
        self.assertNotIn('parameters', tmpl.keys())

    def test_invalid_hot_version(self):
        """Test HOT version check.

        Pass an invalid HOT version to template.Template.__new__() and
        validate that we get a ValueError.
        """
        tmpl_str = "heat_template_version: this-ain't-valid"
        hot_tmpl = template_format.parse(tmpl_str)
        self.assertRaises(exception.InvalidTemplateVersion, template.Template, hot_tmpl)

    def test_valid_hot_version(self):
        """Test HOT version check.

        Pass a valid HOT version to template.Template.__new__() and
        validate that we get back a parsed template.
        """
        tmpl_str = 'heat_template_version: 2013-05-23'
        hot_tmpl = template_format.parse(tmpl_str)
        parsed_tmpl = template.Template(hot_tmpl)
        expected = ('heat_template_version', '2013-05-23')
        observed = parsed_tmpl.version
        self.assertEqual(expected, observed)

    def test_resource_facade(self):
        metadata_snippet = {'resource_facade': 'metadata'}
        deletion_policy_snippet = {'resource_facade': 'deletion_policy'}
        update_policy_snippet = {'resource_facade': 'update_policy'}
        parent_resource = DummyClass()
        parent_resource.metadata_set({'foo': 'bar'})
        parent_resource.t = rsrc_defn.ResourceDefinition('parent', 'SomeType', deletion_policy=rsrc_defn.ResourceDefinition.RETAIN, update_policy={'blarg': 'wibble'})
        tmpl = copy.deepcopy(hot_tpl_empty)
        tmpl['resources'] = {'parent': parent_resource.t.render_hot()}
        parent_resource.stack = parser.Stack(utils.dummy_context(), 'toplevel_stack', template.Template(tmpl))
        parent_resource.stack._resources = {'parent': parent_resource}
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(hot_tpl_empty), parent_resource='parent')
        stack.set_parent_stack(parent_resource.stack)
        self.assertEqual({'foo': 'bar'}, self.resolve(metadata_snippet, stack.t, stack))
        self.assertEqual('Retain', self.resolve(deletion_policy_snippet, stack.t, stack))
        self.assertEqual({'blarg': 'wibble'}, self.resolve(update_policy_snippet, stack.t, stack))

    def test_resource_facade_function(self):
        deletion_policy_snippet = {'resource_facade': 'deletion_policy'}
        parent_resource = DummyClass()
        parent_resource.metadata_set({'foo': 'bar'})
        del_policy = hot_functions.Join(None, 'list_join', ['eta', ['R', 'in']])
        parent_resource.t = rsrc_defn.ResourceDefinition('parent', 'SomeType', deletion_policy=del_policy)
        tmpl = copy.deepcopy(hot_juno_tpl_empty)
        tmpl['resources'] = {'parent': parent_resource.t.render_hot()}
        parent_resource.stack = parser.Stack(utils.dummy_context(), 'toplevel_stack', template.Template(tmpl))
        parent_resource.stack._resources = {'parent': parent_resource}
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(hot_tpl_empty), parent_resource='parent')
        stack.set_parent_stack(parent_resource.stack)
        self.assertEqual('Retain', self.resolve(deletion_policy_snippet, stack.t, stack))

    def test_resource_facade_invalid_arg(self):
        snippet = {'resource_facade': 'wibble'}
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(hot_tpl_empty))
        error = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, stack.t, stack)
        self.assertIn(next(iter(snippet)), str(error))

    def test_resource_facade_missing_deletion_policy(self):
        snippet = {'resource_facade': 'deletion_policy'}
        parent_resource = DummyClass()
        parent_resource.metadata_set({'foo': 'bar'})
        parent_resource.t = rsrc_defn.ResourceDefinition('parent', 'SomeType')
        tmpl = copy.deepcopy(hot_tpl_empty)
        tmpl['resources'] = {'parent': parent_resource.t.render_hot()}
        parent_stack = parser.Stack(utils.dummy_context(), 'toplevel_stack', template.Template(tmpl))
        parent_stack._resources = {'parent': parent_resource}
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(hot_tpl_empty), parent_resource='parent')
        stack.set_parent_stack(parent_stack)
        self.assertEqual('Delete', self.resolve(snippet, stack.t, stack))

    def test_removed_function(self):
        snippet = {'Fn::GetAZs': ''}
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(hot_juno_tpl_empty))
        regxp = 'Fn::GetAZs: The template version is invalid'
        self.assertRaisesRegex(exception.StackValidationFailed, regxp, function.validate, stack.t.parse(stack.defn, snippet))

    def test_add_resource(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          resource1:\n            type: AWS::EC2::Instance\n            properties:\n              property1: value1\n            metadata:\n              foo: bar\n            depends_on:\n              - dummy\n            deletion_policy: Retain\n            update_policy:\n              foo: bar\n          resource2:\n            type: AWS::EC2::Instance\n          resource3:\n            type: AWS::EC2::Instance\n            depends_on:\n              - resource1\n              - dummy\n              - resource2\n        ')
        source = template.Template(hot_tpl)
        empty = template.Template(copy.deepcopy(hot_tpl_empty))
        stack = parser.Stack(utils.dummy_context(), 'test_stack', source)
        for rname, defn in sorted(source.resource_definitions(stack).items()):
            empty.add_resource(defn)
        expected = copy.deepcopy(hot_tpl['resources'])
        expected['resource1']['depends_on'] = []
        expected['resource3']['depends_on'] = ['resource1', 'resource2']
        self.assertEqual(expected, empty.t['resources'])

    def test_add_output(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        outputs:\n          output1:\n            description: An output\n            value: bar\n        ')
        source = template.Template(hot_tpl)
        empty = template.Template(copy.deepcopy(hot_tpl_empty))
        stack = parser.Stack(utils.dummy_context(), 'test_stack', source)
        for defn in source.outputs(stack).values():
            empty.add_output(defn)
        self.assertEqual(hot_tpl['outputs'], empty.t['outputs'])

    def test_filter(self):
        snippet = {'filter': [[None], [1, None, 4, 2, None]]}
        tmpl = template.Template(hot_ocata_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        resolved = self.resolve(snippet, tmpl, stack=stack)
        self.assertEqual([1, 4, 2], resolved)

    def test_filter_wrong_args_type(self):
        snippet = {'filter': 'foo'}
        tmpl = template.Template(hot_ocata_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl, stack=stack)

    def test_filter_wrong_args_number(self):
        snippet = {'filter': [[None], [1, 2], 'foo']}
        tmpl = template.Template(hot_ocata_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl, stack=stack)

    def test_filter_dict(self):
        snippet = {'filter': [[None], {'a': 1}]}
        tmpl = template.Template(hot_ocata_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        self.assertRaises(TypeError, self.resolve, snippet, tmpl, stack=stack)

    def test_filter_str(self):
        snippet = {'filter': [['a'], 'abcd']}
        tmpl = template.Template(hot_ocata_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        self.assertRaises(TypeError, self.resolve, snippet, tmpl, stack=stack)

    def test_filter_str_values(self):
        snippet = {'filter': ['abcd', ['a', 'b', 'c', 'd']]}
        tmpl = template.Template(hot_ocata_tpl_empty)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        self.assertRaises(TypeError, self.resolve, snippet, tmpl, stack=stack)

    def test_make_url_basic(self):
        snippet = {'make_url': {'scheme': 'http', 'host': 'example.com', 'path': '/foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        func = tmpl.parse(None, snippet)
        function.validate(func)
        resolved = function.resolve(func)
        self.assertEqual('http://example.com/foo/bar', resolved)

    def test_make_url_ipv6(self):
        snippet = {'make_url': {'scheme': 'http', 'host': '::1', 'path': '/foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('http://[::1]/foo/bar', resolved)

    def test_make_url_ipv6_ready(self):
        snippet = {'make_url': {'scheme': 'http', 'host': '[::1]', 'path': '/foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('http://[::1]/foo/bar', resolved)

    def test_make_url_port_string(self):
        snippet = {'make_url': {'scheme': 'https', 'host': 'example.com', 'port': '80', 'path': '/foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('https://example.com:80/foo/bar', resolved)

    def test_make_url_port_int(self):
        snippet = {'make_url': {'scheme': 'https', 'host': 'example.com', 'port': 80, 'path': '/foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('https://example.com:80/foo/bar', resolved)

    def test_make_url_port_invalid_high(self):
        snippet = {'make_url': {'scheme': 'https', 'host': 'example.com', 'port': 100000, 'path': '/foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        self.assertRaises(ValueError, self.resolve, snippet, tmpl)

    def test_make_url_port_invalid_low(self):
        snippet = {'make_url': {'scheme': 'https', 'host': 'example.com', 'port': '0', 'path': '/foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        self.assertRaises(ValueError, self.resolve, snippet, tmpl)

    def test_make_url_port_invalid_string(self):
        snippet = {'make_url': {'scheme': 'https', 'host': 'example.com', 'port': '1.1', 'path': '/foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        self.assertRaises(ValueError, self.resolve, snippet, tmpl)

    def test_make_url_username(self):
        snippet = {'make_url': {'scheme': 'http', 'username': 'wibble', 'host': 'example.com', 'path': '/foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('http://wibble@example.com/foo/bar', resolved)

    def test_make_url_username_password(self):
        snippet = {'make_url': {'scheme': 'http', 'username': 'wibble', 'password': 'blarg', 'host': 'example.com', 'path': '/foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('http://wibble:blarg@example.com/foo/bar', resolved)

    def test_make_url_query(self):
        snippet = {'make_url': {'scheme': 'http', 'host': 'example.com', 'path': '/foo/?bar', 'query': {'foo#': 'bar & baz', 'blarg': '/wib=ble/'}}}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertIn(resolved, ['http://example.com/foo/%3Fbar?foo%23=bar+%26+baz&blarg=/wib%3Dble/', 'http://example.com/foo/%3Fbar?blarg=/wib%3Dble/&foo%23=bar+%26+baz'])

    def test_make_url_fragment(self):
        snippet = {'make_url': {'scheme': 'http', 'host': 'example.com', 'path': 'foo/bar', 'fragment': 'baz'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('http://example.com/foo/bar#baz', resolved)

    def test_make_url_file(self):
        snippet = {'make_url': {'scheme': 'file', 'path': 'foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('file:///foo/bar', resolved)

    def test_make_url_file_leading_slash(self):
        snippet = {'make_url': {'scheme': 'file', 'path': '/foo/bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual('file:///foo/bar', resolved)

    def test_make_url_bad_args_type(self):
        snippet = {'make_url': 'http://example.com/foo/bar'}
        tmpl = template.Template(hot_pike_tpl_empty)
        func = tmpl.parse(None, snippet)
        self.assertRaises(exception.StackValidationFailed, function.validate, func)

    def test_make_url_invalid_key(self):
        snippet = {'make_url': {'scheme': 'http', 'host': 'example.com', 'foo': 'bar'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        func = tmpl.parse(None, snippet)
        self.assertRaises(exception.StackValidationFailed, function.validate, func)

    def test_depends_condition(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2016-10-14\n        resources:\n          one:\n            type: OS::Heat::None\n          two:\n            type: OS::Heat::None\n            condition: False\n          three:\n            type: OS::Heat::None\n            depends_on: two\n        ')
        tmpl = template.Template(hot_tpl)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
        stack.validate()
        self.assertEqual({'one', 'three'}, set(stack.resources))

    def test_list_concat(self):
        snippet = {'list_concat': [['v1', 'v2'], ['v3', 'v4']]}
        snippet_resolved = ['v1', 'v2', 'v3', 'v4']
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual(snippet_resolved, resolved)

    def test_list_concat_none(self):
        snippet = {'list_concat': [['v1', 'v2'], ['v3', 'v4'], None]}
        snippet_resolved = ['v1', 'v2', 'v3', 'v4']
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual(snippet_resolved, resolved)

    def test_list_concat_repeat_dict_item(self):
        snippet = {'list_concat': [[{'v1': 'v2'}], [{'v1': 'v2'}]]}
        snippet_resolved = [{'v1': 'v2'}, {'v1': 'v2'}]
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual(snippet_resolved, resolved)

    def test_list_concat_repeat_item(self):
        snippet = {'list_concat': [['v1', 'v2'], ['v2', 'v3']]}
        snippet_resolved = ['v1', 'v2', 'v2', 'v3']
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual(snippet_resolved, resolved)

    def test_list_concat_unique_dict_item(self):
        snippet = {'list_concat_unique': [[{'v1': 'v2'}], [{'v1': 'v2'}]]}
        snippet_resolved = [{'v1': 'v2'}]
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual(snippet_resolved, resolved)

    def test_list_concat_unique(self):
        snippet = {'list_concat_unique': [['v1', 'v2'], ['v1', 'v3']]}
        snippet_resolved = ['v1', 'v2', 'v3']
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertEqual(snippet_resolved, resolved)

    def _test_list_concat_invalid(self, snippet):
        tmpl = template.Template(hot_pike_tpl_empty)
        msg = 'Incorrect arguments'
        exc = self.assertRaises(TypeError, self.resolve, snippet, tmpl)
        self.assertIn(msg, str(exc))

    def test_list_concat_with_dict_arg(self):
        snippet = {'list_concat': [{'k1': 'v2'}, ['v3', 'v4']]}
        self._test_list_concat_invalid(snippet)

    def test_list_concat_with_string_arg(self):
        snippet = {'list_concat': 'I am string'}
        self._test_list_concat_invalid(snippet)

    def test_list_concat_with_string_item(self):
        snippet = {'list_concat': ['v1', 'v2']}
        self._test_list_concat_invalid(snippet)

    def test_contains_with_list(self):
        snippet = {'contains': ['v1', ['v1', 'v2']]}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertTrue(resolved)

    def test_contains_with_string(self):
        snippet = {'contains': ['a', 'abc']}
        tmpl = template.Template(hot_pike_tpl_empty)
        resolved = self.resolve(snippet, tmpl)
        self.assertTrue(resolved)

    def test_contains_with_invalid_args_type(self):
        snippet = {'contains': {'key': 'value'}}
        tmpl = template.Template(hot_pike_tpl_empty)
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
        msg = 'Incorrect arguments to '
        self.assertIn(msg, str(exc))

    def test_contains_with_invalid_args_number(self):
        snippet = {'contains': ['v1', ['v1', 'v2'], 'redundant']}
        tmpl = template.Template(hot_pike_tpl_empty)
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
        msg = 'must be of the form: [value1, [value1, value2]]'
        self.assertIn(msg, str(exc))

    def test_contains_with_invalid_sequence(self):
        snippet = {'contains': ['v1', {'key': 'value'}]}
        tmpl = template.Template(hot_pike_tpl_empty)
        exc = self.assertRaises(TypeError, self.resolve, snippet, tmpl)
        msg = 'should be a sequence'
        self.assertIn(msg, str(exc))