from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import template
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import template_versions
class TestTemplateFunctionList(TestTemplate):
    defaults = [{'functions': 'func1', 'description': 'Function 1'}, {'functions': 'func2', 'description': 'Function 2'}, {'functions': 'condition func', 'description': 'Condition Function'}]

    def setUp(self):
        super(TestTemplateFunctionList, self).setUp()
        self.tv1 = template_versions.TemplateVersion(None, self.defaults[0])
        self.tv2 = template_versions.TemplateVersion(None, self.defaults[1])
        self.tv_with_cf = template_versions.TemplateVersion(None, self.defaults[2])
        self.cmd = template.FunctionList(self.app, None)

    def test_function_list(self):
        arglist = ['version1']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.template_versions.get.return_value = [self.tv1, self.tv2]
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(['Functions', 'Description'], columns)
        self.assertEqual([('func1', 'Function 1'), ('func2', 'Function 2')], list(data))

    def test_function_list_with_condition_func(self):
        arglist = ['version1', '--with_conditions']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.template_versions.get.return_value = [self.tv1, self.tv2, self.tv_with_cf]
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(['Functions', 'Description'], columns)
        self.assertEqual([('func1', 'Function 1'), ('func2', 'Function 2'), ('condition func', 'Condition Function')], list(data))

    def test_function_list_not_found(self):
        arglist = ['bad_version']
        self.template_versions.get.side_effect = exc.HTTPNotFound
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)