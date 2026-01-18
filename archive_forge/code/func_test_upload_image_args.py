import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import volume
from openstack import exceptions
from openstack.tests.unit import base
@mock.patch('openstack.utils.require_microversion', autospec=True, side_effect=[None])
def test_upload_image_args(self, mv_mock):
    sot = volume.Volume(**VOLUME)
    self.resp = mock.Mock()
    self.resp.body = {'os-volume_upload_image': {'a': 'b'}}
    self.resp.status_code = 200
    self.resp.json = mock.Mock(return_value=self.resp.body)
    self.sess.post = mock.Mock(return_value=self.resp)
    self.assertDictEqual({'a': 'b'}, sot.upload_to_image(self.sess, '1', disk_format='2', container_format='3', visibility='4', protected='5'))
    url = 'volumes/%s/action' % FAKE_ID
    body = {'os-volume_upload_image': {'image_name': '1', 'force': False, 'disk_format': '2', 'container_format': '3', 'visibility': '4', 'protected': '5'}}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)
    mv_mock.assert_called_with(self.sess, '3.1')