import copy
import json
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_deploy_template
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalDeployTemplateDelete(TestBaremetalDeployTemplate):

    def setUp(self):
        super(TestBaremetalDeployTemplateDelete, self).setUp()
        self.cmd = baremetal_deploy_template.DeleteBaremetalDeployTemplate(self.app, None)

    def test_baremetal_deploy_template_delete(self):
        arglist = ['zzz-zzzzzz-zzzz']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = 'zzz-zzzzzz-zzzz'
        self.baremetal_mock.deploy_template.delete.assert_called_with(args)

    def test_baremetal_deploy_template_delete_multiple(self):
        arglist = ['zzz-zzzzzz-zzzz', 'fakename']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = ['zzz-zzzzzz-zzzz', 'fakename']
        self.baremetal_mock.deploy_template.delete.assert_has_calls([mock.call(x) for x in args])
        self.assertEqual(2, self.baremetal_mock.deploy_template.delete.call_count)

    def test_baremetal_deploy_template_delete_multiple_with_fail(self):
        arglist = ['zzz-zzzzzz-zzzz', 'badname']
        verifylist = []
        self.baremetal_mock.deploy_template.delete.side_effect = ['', exc.ClientException]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.ClientException, self.cmd.take_action, parsed_args)
        args = ['zzz-zzzzzz-zzzz', 'badname']
        self.baremetal_mock.deploy_template.delete.assert_has_calls([mock.call(x) for x in args])
        self.assertEqual(2, self.baremetal_mock.deploy_template.delete.call_count)

    def test_baremetal_deploy_template_delete_no_template(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)