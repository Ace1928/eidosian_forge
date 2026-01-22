from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import snapshot
from openstack.tests.unit import base
class TestSnapshotActions(base.TestCase):

    def setUp(self):
        super(TestSnapshotActions, self).setUp()
        self.resp = mock.Mock()
        self.resp.body = None
        self.resp.json = mock.Mock(return_value=self.resp.body)
        self.resp.headers = {}
        self.resp.status_code = 202
        self.sess = mock.Mock(spec=adapter.Adapter)
        self.sess.get = mock.Mock()
        self.sess.post = mock.Mock(return_value=self.resp)
        self.sess.default_microversion = None

    def test_reset(self):
        sot = snapshot.Snapshot(**SNAPSHOT)
        self.assertIsNone(sot.reset(self.sess, 'new_status'))
        url = 'snapshots/%s/action' % FAKE_ID
        body = {'os-reset_status': {'status': 'new_status'}}
        self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)