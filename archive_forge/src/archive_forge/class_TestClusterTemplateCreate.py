import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
class TestClusterTemplateCreate(TestClusterTemplate):

    def setUp(self):
        super(TestClusterTemplateCreate, self).setUp()
        attr = dict()
        attr['name'] = 'fake-ct-1'
        self.new_ct = magnum_fakes.FakeClusterTemplate.create_one_cluster_template(attr)
        self.cluster_templates_mock.create = mock.Mock()
        self.cluster_templates_mock.create.return_value = self.new_ct
        self.cluster_templates_mock.get = mock.Mock()
        self.cluster_templates_mock.get.return_value = copy.deepcopy(self.new_ct)
        self.cluster_templates_mock.update = mock.Mock()
        self.cluster_templates_mock.update.return_value = self.new_ct
        self.cmd = osc_ct.CreateClusterTemplate(self.app, None)
        self.data = tuple(map(lambda x: getattr(self.new_ct, x), osc_ct.CLUSTER_TEMPLATE_ATTRIBUTES))

    def test_cluster_template_create_required_args_pass(self):
        """Verifies required arguments."""
        arglist = ['--coe', self.new_ct.coe, '--external-network', self.new_ct.external_network_id, '--image', self.new_ct.image_id, self.new_ct.name]
        verifylist = [('coe', self.new_ct.coe), ('external_network', self.new_ct.external_network_id), ('image', self.new_ct.image_id), ('name', self.new_ct.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cluster_templates_mock.create.assert_called_with(**self.default_create_args)

    def test_cluster_template_create_missing_required_arg(self):
        """Verifies missing required arguments."""
        arglist = ['--external-network', self.new_ct.external_network_id, '--image', self.new_ct.image_id]
        verifylist = [('external_network', self.new_ct.external_network_id), ('image', self.new_ct.image_id)]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)
        arglist.append('--coe')
        arglist.append(self.new_ct.coe)
        verifylist.append(('coe', self.new_ct.coe))
        arglist.remove('--image')
        arglist.remove(self.new_ct.image_id)
        verifylist.remove(('image', self.new_ct.image_id))
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)
        arglist.remove('--external-network')
        arglist.remove(self.new_ct.external_network_id)
        verifylist.remove(('external_network', self.new_ct.external_network_id))
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_cluster_template_create_floating_ips(self):
        """Verifies floating ip parameters."""
        arglist = ['--coe', self.new_ct.coe, '--external-network', self.new_ct.external_network_id, '--image', self.new_ct.image_id, '--floating-ip-enabled', self.new_ct.name]
        verifylist = [('coe', self.new_ct.coe), ('external_network', self.new_ct.external_network_id), ('image', self.new_ct.image_id), ('floating_ip_enabled', [True]), ('name', self.new_ct.name)]
        self.default_create_args['floating_ip_enabled'] = True
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.default_create_args.pop('floating_ip_enabled')
        arglist.append('--floating-ip-disabled')
        verifylist.remove(('floating_ip_enabled', [True]))
        verifylist.append(('floating_ip_enabled', [True, False]))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(InvalidAttribute, self.cmd.take_action, parsed_args)