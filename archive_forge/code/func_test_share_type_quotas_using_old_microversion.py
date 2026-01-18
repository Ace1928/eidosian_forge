from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.v2 import quotas
@ddt.data('get', 'update', 'delete')
def test_share_type_quotas_using_old_microversion(self, operation):
    manager = self._get_manager('2.38')
    with mock.patch.object(manager, '_%s' % operation, mock.Mock(return_value='fake_delete')):
        self.assertRaises(TypeError, getattr(manager, operation), 'fake_tenant_id', share_type='fake_share_type')
        getattr(manager, '_%s' % operation).assert_not_called()