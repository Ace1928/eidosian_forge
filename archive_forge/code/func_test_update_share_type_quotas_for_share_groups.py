from random import randint
import ddt
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data('--share-groups', '--share-group-snapshots')
@utils.skip_if_microversion_not_supported('2.40')
def test_update_share_type_quotas_for_share_groups(self, arg):
    cmd = 'quota-update %s --share-type %s %s 13' % (self.project_id, self.st_id, arg)
    self.assertRaises(exceptions.CommandFailed, self.admin_client.manila, cmd, microversion='2.40')