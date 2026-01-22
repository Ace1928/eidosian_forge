from unittest import mock
from keystoneauth1 import adapter
from openstack.shared_file_system.v2 import share
from openstack.tests.unit import base
class TestShareActions(TestShares):

    def setUp(self):
        super().setUp()
        self.resp = mock.Mock()
        self.resp.body = None
        self.resp.status_code = 202
        self.resp.json = mock.Mock(return_value=self.resp.body)
        self.sess = mock.Mock(spec=adapter.Adapter)
        self.sess.default_microversion = '3.0'
        self.sess.post = mock.Mock(return_value=self.resp)
        self.sess._get_connection = mock.Mock(return_value=self.cloud)

    def test_shrink_share(self):
        sot = share.Share(**EXAMPLE)
        microversion = sot._get_microversion(self.sess, action='patch')
        self.assertIsNone(sot.shrink_share(self.sess, new_size=1))
        url = f'shares/{IDENTIFIER}/action'
        body = {'shrink': {'new_size': 1}}
        headers = {'Accept': ''}
        self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=microversion)

    def test_extend_share(self):
        sot = share.Share(**EXAMPLE)
        microversion = sot._get_microversion(self.sess, action='patch')
        self.assertIsNone(sot.extend_share(self.sess, new_size=3))
        url = f'shares/{IDENTIFIER}/action'
        body = {'extend': {'new_size': 3}}
        headers = {'Accept': ''}
        self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=microversion)

    def test_revert_to_snapshot(self):
        sot = share.Share(**EXAMPLE)
        microversion = sot._get_microversion(self.sess, action='patch')
        self.assertIsNone(sot.revert_to_snapshot(self.sess, 'fake_id'))
        url = f'shares/{IDENTIFIER}/action'
        body = {'revert': {'snapshot_id': 'fake_id'}}
        headers = {'Accept': ''}
        self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=microversion)

    def test_manage_share(self):
        sot = share.Share()
        self.resp.headers = {}
        self.resp.json = mock.Mock(return_value={'share': {'name': 'test_share', 'size': 1}})
        export_path = '10.254.0 .5:/shares/share-42033c24-0261-424f-abda-4fef2f6dbfd5.'
        params = {'name': 'test_share'}
        res = sot.manage(self.sess, sot['share_protocol'], export_path, sot['host'], **params)
        self.assertEqual(res.name, 'test_share')
        self.assertEqual(res.size, 1)
        jsonDict = {'share': {'protocol': sot['share_protocol'], 'export_path': export_path, 'service_host': sot['host'], 'name': 'test_share'}}
        self.sess.post.assert_called_once_with('shares/manage', json=jsonDict)

    def test_unmanage_share(self):
        sot = share.Share(**EXAMPLE)
        microversion = sot._get_microversion(self.sess, action='patch')
        self.assertIsNone(sot.unmanage(self.sess))
        url = 'shares/%s/action' % IDENTIFIER
        body = {'unmanage': None}
        self.sess.post.assert_called_with(url, json=body, headers={'Accept': ''}, microversion=microversion)