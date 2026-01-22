from unittest import mock
from keystoneauth1 import adapter
from openstack.shared_file_system.v2 import share_instance
from openstack.tests.unit import base
class TestShareInstanceActions(TestShareInstances):

    def setUp(self):
        super(TestShareInstanceActions, self).setUp()
        self.resp = mock.Mock()
        self.resp.body = None
        self.resp.status_code = 200
        self.resp.json = mock.Mock(return_value=self.resp.body)
        self.sess = mock.Mock(spec=adapter.Adapter)
        self.sess.default_microversion = '3.0'
        self.sess.post = mock.Mock(return_value=self.resp)
        self.sess._get_connection = mock.Mock(return_value=self.cloud)

    def test_reset_status(self):
        sot = share_instance.ShareInstance(**EXAMPLE)
        microversion = sot._get_microversion(self.sess, action='patch')
        self.assertIsNone(sot.reset_status(self.sess, 'active'))
        url = f'share_instances/{IDENTIFIER}/action'
        body = {'reset_status': {'status': 'active'}}
        headers = {'Accept': ''}
        self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=microversion)

    def test_force_delete(self):
        sot = share_instance.ShareInstance(**EXAMPLE)
        microversion = sot._get_microversion(self.sess, action='delete')
        self.assertIsNone(sot.force_delete(self.sess))
        url = f'share_instances/{IDENTIFIER}/action'
        body = {'force_delete': None}
        headers = {'Accept': ''}
        self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=microversion)