import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import quota
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestQuotaDefaultsShow(TestQuota):
    qt_defaults = {'health_monitor': 1, 'listener': 2, 'load_balancer': 3, 'member': 4, 'pool': 5, 'l7policy': 6, 'l7rule': 7}

    def setUp(self):
        super().setUp()
        self.api_mock.quota_defaults_show.return_value = {'quota': self.qt_defaults}
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = quota.ShowQuotaDefaults(self.app, None)

    def test_quota_defaults_show(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        rows, data = self.cmd.take_action(parsed_args)
        data = dict(zip(rows, data))
        self.api_mock.quota_defaults_show.assert_called_with()
        self.assertEqual(self.qt_defaults, data)