import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
class TestCreateNetworkLog(TestNetworkLog):

    def setUp(self):
        super(TestCreateNetworkLog, self).setUp()
        self.neutronclient.create_network_log = mock.Mock(return_value={'log': _log})
        self.mocked = self.neutronclient.create_network_log
        self.cmd = network_log.CreateNetworkLog(self.app, self.namespace)
        loggables = {'loggable_resources': [{'type': RES_TYPE_SG}, {'type': RES_TYPE_FWG}]}
        self.neutronclient.list_network_loggable_resources = mock.Mock(return_value=loggables)

    def _update_expect_response(self, request, response):
        """Set expected request and response

        :param request
            A dictionary of request body(dict of verifylist)
        :param response
            A OrderedDict of request body
        """
        self.neutronclient.create_network_log.return_value = {'log': dict(response)}
        osc_utils.find_project.return_value.id = response['project_id']
        self.data = _generate_data(ordered_dict=response)
        self.ordered_data = tuple((response[column] for column in self.ordered_columns))

    def _set_all_params(self, args={}):
        name = args.get('name', 'my-log')
        desc = args.get('description', 'my-description-for-log')
        event = args.get('event', 'ACCEPT')
        resource = args.get('resource', 'id-target-log')
        target = args.get('target', 'id-target-log')
        resource_type = args.get('resource_type', 'security_group')
        project = args.get('project_id', 'id-my-project')
        arglist = [name, '--description', desc, '--enable', '--target', target, '--resource', resource, '--event', event, '--project', project, '--resource-type', resource_type]
        verifylist = [('description', desc), ('enable', True), ('event', event), ('name', name), ('target', target), ('project', project), ('resource', target), ('resource_type', resource_type)]
        return (arglist, verifylist)

    def _test_create_with_all_params(self, args={}):
        arglist, verifylist = self._set_all_params(args)
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.check_results(headers, data, request)

    def test_create_with_no_options_and_raise(self):
        arglist = []
        verifylist = []
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_mandatory_params(self):
        name = self.res['name']
        arglist = [name, '--resource-type', RES_TYPE_SG]
        verifylist = [('name', name), ('resource_type', RES_TYPE_SG)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        expect = {'name': self.res['name'], 'resource_type': self.res['resource_type']}
        self.mocked.assert_called_once_with({'log': expect})
        self.assertEqual(self.ordered_headers, headers)
        self.assertEqual(self.ordered_data, data)

    def test_create_with_disable(self):
        name = self.res['name']
        arglist = [name, '--resource-type', RES_TYPE_SG, '--disable']
        verifylist = [('name', name), ('resource_type', RES_TYPE_SG), ('disable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        expect = {'name': self.res['name'], 'resource_type': self.res['resource_type'], 'enabled': False}
        self.mocked.assert_called_once_with({'log': expect})
        self.assertEqual(self.ordered_headers, headers)
        self.assertEqual(self.ordered_data, data)

    def test_create_with_all_params(self):
        self._test_create_with_all_params()

    def test_create_with_all_params_event_drop(self):
        self._test_create_with_all_params({'event': 'DROP'})

    def test_create_with_all_params_event_all(self):
        self._test_create_with_all_params({'event': 'ALL'})

    def test_create_with_all_params_except_event(self):
        arglist, verifylist = self._set_all_params({'event': ''})
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_all_params_event_upper_capitalized(self):
        for event in ('all', 'All', 'dROP', 'accePt', 'accept', 'drop'):
            arglist, verifylist = self._set_all_params({'event': event})
            self.assertRaises(testtools.matchers._impl.MismatchError, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_all_params_resource_type_upper_capitalized(self):
        for res_type in ('SECURITY_GROUP', 'Security_group', 'security_Group'):
            arglist, verifylist = self._set_all_params({'resource_type': res_type})
            self.assertRaises(testtools.matchers._impl.MismatchError, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_valid_fwg_resource(self):
        name = self.res['name']
        resource_id = 'valid_fwg_id'
        resource_type = RES_TYPE_FWG
        with mock.patch.object(self.neutronclient, 'find_resource', return_value={'id': resource_id}):
            arglist = [name, '--resource-type', resource_type, '--resource', resource_id]
            verifylist = [('name', name), ('resource_type', resource_type), ('resource', resource_id)]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            headers, data = self.cmd.take_action(parsed_args)
            expect = {'name': self.res['name'], 'resource_type': RES_TYPE_FWG, 'resource_id': 'valid_fwg_id'}
            self.neutronclient.find_resource.assert_called_with(resource_type, resource_id, cmd_resource='fwaas_firewall_group')
            self.mocked.assert_called_once_with({'log': expect})
            self.assertEqual(self.ordered_headers, headers)
            self.assertEqual(self.ordered_data, data)

    def test_create_with_invalid_fwg_resource(self):
        name = self.res['name']
        resource_id = 'invalid_fwg_id'
        resource_type = RES_TYPE_FWG
        with mock.patch.object(self.neutronclient, 'find_resource', side_effect=exceptions.NotFound(code=0)):
            arglist = [name, '--resource-type', resource_type, '--resource', resource_id]
            verifylist = [('name', name), ('resource_type', resource_type), ('resource', resource_id)]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            self.assertRaises(exceptions.NotFound, self.cmd.take_action, parsed_args)
            self.neutronclient.find_resource.assert_called_with(resource_type, resource_id, cmd_resource='fwaas_firewall_group')
            self.mocked.assert_not_called()

    def test_create_with_invalid_resource_type(self):
        name = self.res['name']
        resource_type = 'invalid_resource_type'
        resource_id = 'valid_fwg_id'
        with mock.patch.object(self.neutronclient, 'find_resource', side_effect=exceptions.NotFound(code=0)):
            arglist = [name, '--resource-type', resource_type, '--resource', resource_id]
            verifylist = [('name', name), ('resource_type', resource_type), ('resource', resource_id)]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            self.assertRaises(exceptions.NotFound, self.cmd.take_action, parsed_args)
            self.neutronclient.find_resource.assert_called_with(resource_type, resource_id, cmd_resource=None)
            self.mocked.assert_not_called()