from random import randint
import ddt
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data(*set(['2.39', '2.40', REPLICA_QUOTAS_MICROVERSION, api_versions.MAX_VERSION]))
def test_update_share_type_quotas_positive(self, microversion):
    if not utils.is_microversion_supported(microversion):
        msg = "Microversion '%s' not supported." % microversion
        raise self.skipException(msg)
    cmd = 'quota-show --tenant-id %s ' % self.project_id
    quotas_raw = self.admin_client.manila(cmd, microversion=microversion)
    p_quotas = output_parser.details(quotas_raw)
    st_custom_quotas = {'shares': _get_share_type_quota_values(p_quotas['shares']), 'snapshots': _get_share_type_quota_values(p_quotas['snapshots']), 'gigabytes': _get_share_type_quota_values(p_quotas['gigabytes']), 'snapshot_gigabytes': _get_share_type_quota_values(p_quotas['snapshot_gigabytes'])}
    supports_share_replica_quotas = api_versions.APIVersion(microversion) >= api_versions.APIVersion(REPLICA_QUOTAS_MICROVERSION)
    if supports_share_replica_quotas:
        st_custom_quotas['share_replicas'] = _get_share_type_quota_values(p_quotas['share_replicas'])
        st_custom_quotas['replica_gigabytes'] = _get_share_type_quota_values(p_quotas['replica_gigabytes'])
        replica_params = ' --share-replicas %s --replica-gigabytes %s' % (st_custom_quotas['share_replicas'], st_custom_quotas['replica_gigabytes'])
    cmd = 'quota-update %s --share-type %s --shares %s --gigabytes %s --snapshots %s --snapshot-gigabytes %s' % (self.project_id, self.st_id, st_custom_quotas['shares'], st_custom_quotas['gigabytes'], st_custom_quotas['snapshots'], st_custom_quotas['snapshot_gigabytes'])
    if supports_share_replica_quotas:
        cmd += replica_params
    self.admin_client.manila(cmd, microversion=microversion)
    self._verify_current_st_quotas_equal_to(st_custom_quotas, microversion)
    cmd = 'quota-delete --tenant-id %s --share-type %s' % (self.project_id, self.st_id)
    self.admin_client.manila(cmd, microversion=microversion)
    self._verify_current_st_quotas_equal_to(p_quotas, microversion)