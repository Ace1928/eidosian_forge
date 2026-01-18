from random import randint
import ddt
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data(*set(['2.40', api_versions.MAX_VERSION]))
def test_update_quotas_for_share_groups(self, microversion):
    if not utils.is_microversion_supported(microversion):
        msg = "Microversion '%s' not supported." % microversion
        raise self.skipException(msg)
    cmd = 'quota-defaults'
    quotas_raw = self.admin_client.manila(cmd, microversion=microversion)
    default_quotas = output_parser.details(quotas_raw)
    cmd = 'quota-show --tenant-id %s ' % self.project_id
    quotas_raw = self.admin_client.manila(cmd, microversion=microversion)
    p_quotas = output_parser.details(quotas_raw)
    p_custom_quotas = {'share_groups': -1 if int(p_quotas['share_groups']) != -1 else 999, 'share_group_snapshots': -1 if int(p_quotas['share_group_snapshots']) != -1 else 999}
    cmd = 'quota-update %s --share-groups %s --share-group-snapshots %s' % (self.project_id, p_custom_quotas['share_groups'], p_custom_quotas['share_group_snapshots'])
    self.admin_client.manila(cmd, microversion=microversion)
    self._verify_current_quotas_equal_to(p_custom_quotas, microversion)
    cmd = 'quota-delete --tenant-id %s --share-type %s' % (self.project_id, self.st_id)
    self.admin_client.manila(cmd, microversion=microversion)
    self._verify_current_quotas_equal_to(default_quotas, microversion)
    cmd = 'quota-update %s --share-groups %s --share-group-snapshots %s' % (self.project_id, p_quotas['share_groups'], p_quotas['share_group_snapshots'])
    self.admin_client.manila(cmd, microversion=microversion)
    self._verify_current_quotas_equal_to(p_quotas, microversion)