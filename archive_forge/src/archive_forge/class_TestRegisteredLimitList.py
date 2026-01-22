import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import registered_limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRegisteredLimitList(TestRegisteredLimit):

    def setUp(self):
        super(TestRegisteredLimitList, self).setUp()
        self.registered_limit_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.REGISTERED_LIMIT), loaded=True)
        self.cmd = registered_limit.ShowRegisteredLimit(self.app, None)

    def test_limit_show(self):
        arglist = [identity_fakes.registered_limit_id]
        verifylist = [('registered_limit_id', identity_fakes.registered_limit_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.registered_limit_mock.get.assert_called_with(identity_fakes.registered_limit_id)
        collist = ('default_limit', 'description', 'id', 'region_id', 'resource_name', 'service_id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.registered_limit_default_limit, None, identity_fakes.registered_limit_id, None, identity_fakes.registered_limit_resource_name, identity_fakes.service_id)
        self.assertEqual(datalist, data)