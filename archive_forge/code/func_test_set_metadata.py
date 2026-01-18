from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import aggregate
from openstack.tests.unit import base
def test_set_metadata(self):
    sot = aggregate.Aggregate(**EXAMPLE)
    sot.set_metadata(self.sess, {'key: value'})
    url = 'os-aggregates/4/action'
    body = {'set_metadata': {'metadata': {'key: value'}}}
    self.sess.post.assert_called_with(url, json=body, microversion=None)