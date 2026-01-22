from unittest import mock
import uuid
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import vpnservice
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestCreateVPNService(TestVPNService, common.TestCreateVPNaaS):

    def setUp(self):
        super(TestCreateVPNService, self).setUp()
        self.networkclient.create_vpn_service = mock.Mock(return_value=_vpnservice)
        self.mocked = self.networkclient.create_vpn_service
        self.cmd = vpnservice.CreateVPNService(self.app, self.namespace)

    def _update_expect_response(self, request, response):
        """Set expected request and response

        :param request
            A dictionary of request body(dict of verifylist)
        :param response
            A OrderedDict of request body
        """
        self.networkclient.create_vpn_service.return_value = response
        osc_utils.find_project.return_value.id = response['project_id']
        self.data = _generate_data(ordered_dict=response)
        self.ordered_data = tuple((response[column] for column in self.ordered_columns))

    def _set_all_params(self):
        name = self.args.get('name')
        description = self.args.get('description')
        router_id = self.args.get('router_id')
        subnet_id = self.args.get('subnet_id')
        project_id = self.args.get('project_id')
        arglist = ['--description', description, '--project', project_id, '--subnet', subnet_id, '--router', router_id, name]
        verifylist = [('description', description), ('project', project_id), ('subnet', subnet_id), ('router', router_id), ('name', name)]
        return (arglist, verifylist)

    def _test_create_with_all_params(self):
        arglist, verifylist = self._set_all_params()
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self._check_results(headers, data, request)

    def test_create_with_all_params(self):
        self._test_create_with_all_params()