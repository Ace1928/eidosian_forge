import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from tempest.lib import exceptions
from novaclient.tests.functional import base
class TestInstanceActionCLIV258(TestInstanceActionCLI):
    """Instance action functional tests for v2.58 nova-api microversion."""
    COMPUTE_API_VERSION = '2.58'
    expect_event_hostId_field = False

    def test_list_instance_action_with_marker_and_limit(self):
        server = self._create_server()
        server.stop()
        output = self.nova('instance-action-list %s --limit 1' % server.id)
        marker_req = self._get_column_value_from_single_row_table(output, 'Request_ID')
        action = self._get_list_of_values_from_single_column_table(output, 'Action')
        self.assertEqual(action, ['stop'])
        output = self.nova('instance-action-list %s --limit 1 --marker %s' % (server.id, marker_req))
        action = self._get_list_of_values_from_single_column_table(output, 'Action')
        self.assertEqual(action, ['create'])
        if not self.expect_event_hostId_field:
            output = self.nova('instance-action %s %s' % (server.id, marker_req))
            self.assertNotIn("'host'", output)
            self.assertNotIn("'hostId'", output)

    def test_list_instance_action_with_changes_since(self):
        before_create = timeutils.utcnow().replace(microsecond=0).isoformat()
        server = self._create_server()
        time.sleep(2)
        before_stop = timeutils.utcnow().replace(microsecond=0).isoformat()
        server.stop()
        create_output = self.nova('instance-action-list %s --changes-since %s' % (server.id, before_create))
        action = self._get_list_of_values_from_single_column_table(create_output, 'Action')
        self.assertEqual(action, ['create', 'stop'])
        stop_output = self.nova('instance-action-list %s --changes-since %s' % (server.id, before_stop))
        action = self._get_list_of_values_from_single_column_table(stop_output, 'Action')
        self.assertEqual(action, ['stop'], 'Expected to find the stop action with --changes-since=%s but got: %s\n\nFirst instance-action-list output: %s' % (before_stop, stop_output, create_output))