import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
class TestNetworkLog(test_fakes.TestNeutronClientOSCV2):

    def check_results(self, headers, data, exp_req, is_list=False):
        if is_list:
            req_body = {'logs': [exp_req]}
        else:
            req_body = {'log': exp_req}
        self.mocked.assert_called_once_with(req_body)
        self.assertEqual(self.ordered_headers, headers)
        self.assertEqual(self.ordered_data, data)

    def setUp(self):
        super(TestNetworkLog, self).setUp()
        self.neutronclient.find_resource = mock.Mock()
        self.neutronclient.find_resource.side_effect = lambda x, y, **k: {'id': y}
        osc_utils.find_project = mock.Mock()
        osc_utils.find_project.id = _log['project_id']
        self.res = _log
        self.headers = ('ID', 'Description', 'Enabled', 'Name', 'Target', 'Project', 'Resource', 'Type', 'Event')
        self.data = _generate_data()
        self.ordered_headers = ('Description', 'Enabled', 'Event', 'ID', 'Name', 'Project', 'Resource', 'Target', 'Type')
        self.ordered_data = (_log['description'], _log['enabled'], _log['event'], _log['id'], _log['name'], _log['project_id'], _log['resource_id'], _log['target_id'], _log['resource_type'])
        self.ordered_columns = ('description', 'enabled', 'event', 'id', 'name', 'project_id', 'resource_id', 'target_id', 'resource_type')