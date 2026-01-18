from unittest import mock
from keystoneauth1 import adapter
from openstack.common import tag
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_remove_all_tags(self):
    res = self.sot
    sess = self.session
    res.tags.extend(['blue_old', 'green_old'])
    result = res.remove_all_tags(sess)
    self.assertEqual([], res.tags)
    self.assertEqual(res, result)
    url = self.base_path + '/' + res.id + '/tags'
    sess.delete.assert_called_once_with(url)