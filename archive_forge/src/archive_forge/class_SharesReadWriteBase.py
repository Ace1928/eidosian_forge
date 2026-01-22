import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.ddt
class SharesReadWriteBase(base.BaseTestCase):
    protocol = None

    def setUp(self):
        super(SharesReadWriteBase, self).setUp()
        if self.protocol not in CONF.enable_protocols:
            message = '%s tests are disabled' % self.protocol
            raise self.skipException(message)
        self.name = data_utils.rand_name('autotest_share_name')
        self.description = data_utils.rand_name('autotest_share_description')
        self.share = self.create_share(share_protocol=self.protocol, size=1, name=self.name, description=self.description, client=self.get_user_client())

    def test_create_delete_share(self):
        name = data_utils.rand_name('autotest_share_name')
        create = self.create_share(self.protocol, name=name, client=self.user_client)
        self.assertEqual('creating', create['status'])
        self.assertEqual(name, create['name'])
        self.assertEqual('1', create['size'])
        self.assertEqual(self.protocol.upper(), create['share_proto'])
        self.user_client.delete_share(create['id'])
        self.user_client.wait_for_share_deletion(create['id'])

    def test_create_update_share(self):
        name = data_utils.rand_name('autotest_share_name')
        new_name = 'new_' + name
        description = data_utils.rand_name('autotest_share_description')
        new_description = 'new_' + description
        create = self.create_share(self.protocol, name=name, description=description, client=self.user_client)
        self.assertEqual(name, create['name'])
        self.assertEqual(description, create['description'])
        self.assertEqual('False', create['is_public'])
        self.user_client.update_share(create['id'], name=new_name, description=new_description)
        get = self.user_client.get_share(create['id'])
        self.assertEqual(new_name, get['name'])
        self.assertEqual(new_description, get['description'])
        self.assertEqual('False', get['is_public'])
        self.admin_client.update_share(create['id'], is_public=True)
        get = self.user_client.get_share(create['id'])
        self.assertEqual(new_name, get['name'])
        self.assertEqual(new_description, get['description'])
        self.assertEqual('True', get['is_public'])

    def test_get_share(self):
        get = self.user_client.get_share(self.share['id'])
        self.assertEqual(self.name, get['name'])
        self.assertEqual(self.description, get['description'])
        self.assertEqual('1', get['size'])
        self.assertEqual(self.protocol.upper(), get['share_proto'])

    def test_create_delete_with_wait(self):
        name = data_utils.rand_name('share-with-wait-%s')
        description = data_utils.rand_name('we-wait-until-share-is-ready')
        share_1, share_2 = (self.create_share(self.protocol, name=name % num, description=description, use_wait_option=True, client=self.user_client) for num in range(0, 2))
        self.assertEqual('available', share_1['status'])
        self.assertEqual('available', share_2['status'])
        self.delete_share([share_1['id'], share_2['id']], wait=True, client=self.user_client)
        for share in (share_1, share_2):
            self.assertRaises(exceptions.NotFound, self.user_client.get_share, share['id'])

    def test_create_soft_delete_and_restore_share(self):
        self.skip_if_microversion_not_supported('2.69')
        microversion = '2.69'
        description = data_utils.rand_name('we-wait-until-share-is-ready')
        share = self.create_share(self.protocol, name='share_name', description=description, use_wait_option=True, client=self.user_client)
        self.assertEqual('available', share['status'])
        self.soft_delete_share([share['id']], client=self.user_client, microversion=microversion)
        self.user_client.wait_for_share_soft_deletion(share['id'])
        result = self.user_client.list_shares(is_soft_deleted=True, microversion=microversion)
        share_ids = [sh['ID'] for sh in result]
        self.assertIn(share['id'], share_ids)
        self.restore_share([share['id']], client=self.user_client, microversion=microversion)
        self.user_client.wait_for_share_restore(share['id'])
        result1 = self.user_client.list_shares(is_soft_deleted=True, microversion=microversion)
        share_ids1 = [sh['ID'] for sh in result1]
        self.assertNotIn(share['id'], share_ids1)