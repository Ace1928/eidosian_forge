from unittest import mock
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import quotas as osc_quotas
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_quota_delete_api_version_exception(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.38')
    arglist = [self.project.id, '--share-type', 'default']
    verifylist = [('project', self.project.id), ('share_type', 'default')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)