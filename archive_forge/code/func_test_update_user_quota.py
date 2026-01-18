from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.v2 import quotas
@ddt.data('2.6', '2.7', '2.38', '2.39', '2.40', REPLICA_QUOTAS_MICROVERSION)
def test_update_user_quota(self, microversion):
    tenant_id = 'test'
    user_id = 'fake_user'
    manager = self._get_manager(microversion)
    resource_path = self._get_resource_path(microversion)
    expected_url = '%s/test?user_id=fake_user' % resource_path
    expected_body = {'quota_set': {'tenant_id': tenant_id, 'shares': 1, 'snapshots': 2, 'gigabytes': 3, 'snapshot_gigabytes': 4, 'share_networks': 5}}
    kwargs = {'shares': expected_body['quota_set']['shares'], 'snapshots': expected_body['quota_set']['snapshots'], 'gigabytes': expected_body['quota_set']['gigabytes'], 'snapshot_gigabytes': expected_body['quota_set']['snapshot_gigabytes'], 'share_networks': expected_body['quota_set']['share_networks'], 'user_id': user_id}
    if api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.40'):
        expected_body['quota_set']['share_groups'] = 6
        expected_body['quota_set']['share_group_snapshots'] = 7
        kwargs['share_groups'] = expected_body['quota_set']['share_groups']
        kwargs['share_group_snapshots'] = expected_body['quota_set']['share_group_snapshots']
    if api_versions.APIVersion(microversion) >= api_versions.APIVersion(REPLICA_QUOTAS_MICROVERSION):
        expected_body['quota_set']['share_replicas'] = 8
        expected_body['quota_set']['replica_gigabytes'] = 9
        kwargs['share_replicas'] = expected_body['quota_set']['share_replicas']
        kwargs['replica_gigabytes'] = expected_body['quota_set']['replica_gigabytes']
    if api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.62'):
        expected_body['quota_set']['per_share_gigabytes'] = 10
        kwargs['per_share_gigabytes'] = expected_body['quota_set']['per_share_gigabytes']
    with mock.patch.object(manager, '_update', mock.Mock(return_value='fake_update')):
        manager.update(tenant_id, **kwargs)
        manager._update.assert_called_once_with(expected_url, expected_body, 'quota_set')