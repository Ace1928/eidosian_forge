from unittest import mock
from keystoneauth1 import adapter
from openstack.common import tag
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_remove_single_tag(self):
    res = self.sot
    sess = self.session
    res.tags.extend(['blue', 'dummy'])
    result = res.remove_tag(sess, 'dummy')
    self.assertEqual(['blue'], res.tags)
    self.assertEqual(res, result)
    url = self.base_path + '/' + res.id + '/tags/dummy'
    sess.delete.assert_called_once_with(url)