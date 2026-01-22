from tempest.lib.common.utils import data_utils
from manilaclient.tests.functional import base
class ShareTransferTests(base.BaseTestCase):
    """Check of base share transfers command"""

    def setUp(self):
        super(ShareTransferTests, self).setUp()
        self.share_type = self.create_share_type(name=data_utils.rand_name('test_share_type'), driver_handles_share_servers=False)

    def test_transfer_create_list_show_delete(self):
        """Create, list, show and delete a share transfer"""
        self.skip_if_microversion_not_supported('2.77')
        share = self.create_share(share_protocol='nfs', size=1, name=data_utils.rand_name('autotest_share_name'), client=self.user_client, share_type=self.share_type['ID'], use_wait_option=True)
        self.assertEqual('available', share['status'])
        transfer = self.create_share_transfer(share['id'], name='test_share_transfer')
        self.assertIn('auth_key', transfer)
        transfers = self.list_share_transfer()
        self.assertTrue(len(transfers) > 0)
        transfer_show = self.get_share_transfer(transfer['id'])
        self.assertEqual(transfer_show['name'], transfer['name'])
        self.assertNotIn('auth_key', transfer_show)
        self.delete_share_transfer(transfer['id'])
        self.user_client.wait_for_transfer_deletion(transfer['id'])
        share = self.user_client.get_share(share['id'])
        self.assertEqual('available', share['status'])
        self.user_client.delete_share(share['id'])
        self.user_client.wait_for_share_deletion(share['id'])

    def test_transfer_accept(self):
        """Show share transfer accept"""
        self.skip_if_microversion_not_supported('2.77')
        share = self.create_share(share_protocol='nfs', size=1, name=data_utils.rand_name('autotest_share_name'), client=self.user_client, share_type=self.share_type['ID'], use_wait_option=True)
        self.assertEqual('available', share['status'])
        transfer = self.create_share_transfer(share['id'], name='test_share_transfer')
        share = self.user_client.get_share(share['id'])
        transfer_id = transfer['id']
        auth_key = transfer['auth_key']
        self.assertEqual('share', transfer['resource_type'])
        self.assertEqual('test_share_transfer', transfer['name'])
        self.assertEqual('awaiting_transfer', share['status'])
        self.accept_share_transfer(transfer_id, auth_key)
        self.user_client.wait_for_transfer_deletion(transfer_id)
        share = self.user_client.get_share(share['id'])
        self.assertEqual('available', share['status'])
        self.user_client.delete_share(share['id'])
        self.user_client.wait_for_share_deletion(share['id'])