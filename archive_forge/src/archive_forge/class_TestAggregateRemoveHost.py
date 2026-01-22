from unittest import mock
from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import aggregate
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestAggregateRemoveHost(TestAggregate):

    def setUp(self):
        super(TestAggregateRemoveHost, self).setUp()
        self.compute_sdk_client.find_aggregate.return_value = self.fake_ag
        self.compute_sdk_client.remove_host_from_aggregate.return_value = self.fake_ag
        self.cmd = aggregate.RemoveAggregateHost(self.app, None)

    def test_aggregate_remove_host(self):
        arglist = ['ag1', 'host1']
        verifylist = [('aggregate', 'ag1'), ('host', 'host1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_aggregate.assert_called_once_with(parsed_args.aggregate, ignore_missing=False)
        self.compute_sdk_client.remove_host_from_aggregate.assert_called_once_with(self.fake_ag.id, parsed_args.host)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)