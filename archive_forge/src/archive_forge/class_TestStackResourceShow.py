import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
class TestStackResourceShow(TestResource):
    response = {'attributes': {}, 'creation_time': '2016-02-01T20:20:53', 'description': 'a resource', 'links': [{'rel': 'stack', 'href': 'http://heat.example.com:8004/my_stack/12'}], 'logical_resource_id': 'my_resource', 'physical_resource_id': '1234', 'required_by': [], 'resource_name': 'my_resource', 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'state changed', 'resource_type': 'OS::Heat::None', 'updated_time': '2016-02-01T20:20:53'}

    def setUp(self):
        super(TestStackResourceShow, self).setUp()
        self.cmd = resource.ResourceShow(self.app, None)
        self.resource_client.get.return_value = v1_resources.Resource(None, self.response)

    def test_resource_show(self):
        arglist = ['my_stack', 'my_resource']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.resource_client.get.assert_called_with('my_stack', 'my_resource', with_attr=None)
        for key in self.response:
            self.assertIn(key, columns)
            self.assertIn(self.response[key], data)

    def test_resource_show_with_attr(self):
        arglist = ['my_stack', 'my_resource', '--with-attr', 'foo', '--with-attr', 'bar']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.resource_client.get.assert_called_with('my_stack', 'my_resource', with_attr=['foo', 'bar'])
        for key in self.response:
            self.assertIn(key, columns)
            self.assertIn(self.response[key], data)

    def test_resource_show_not_found(self):
        arglist = ['my_stack', 'bad_resource']
        self.resource_client.get.side_effect = heat_exc.HTTPNotFound
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Stack or resource not found: my_stack bad_resource', str(error))