import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
class TestResourceMarkUnhealthy(TestResource):

    def setUp(self):
        super(TestResourceMarkUnhealthy, self).setUp()
        self.cmd = resource.ResourceMarkUnhealthy(self.app, None)
        self.resource_client.mark_unhealthy = mock.Mock()

    def test_resource_mark_unhealthy(self):
        arglist = ['my_stack', 'my_resource', 'reason']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.resource_client.mark_unhealthy.assert_called_with(**{'stack_id': 'my_stack', 'resource_name': 'my_resource', 'mark_unhealthy': True, 'resource_status_reason': 'reason'})

    def test_resource_mark_unhealthy_reset(self):
        arglist = ['my_stack', 'my_resource', '--reset']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.resource_client.mark_unhealthy.assert_called_with(**{'stack_id': 'my_stack', 'resource_name': 'my_resource', 'mark_unhealthy': False, 'resource_status_reason': ''})

    def test_resource_mark_unhealthy_not_found(self):
        arglist = ['my_stack', 'my_resource', '--reset']
        self.resource_client.mark_unhealthy.side_effect = heat_exc.HTTPNotFound
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Stack or resource not found: my_stack my_resource', str(error))