from unittest import mock
from openstack.clustering.v1 import cluster
from openstack.tests.unit import base
def test_policy_attach(self):
    sot = cluster.Cluster(**FAKE)
    resp = mock.Mock()
    resp.json = mock.Mock(return_value='')
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=resp)
    params = {'enabled': True}
    self.assertEqual('', sot.policy_attach(sess, 'POLICY', **params))
    url = 'clusters/%s/actions' % sot.id
    body = {'policy_attach': {'policy_id': 'POLICY', 'enabled': True}}
    sess.post.assert_called_once_with(url, json=body)