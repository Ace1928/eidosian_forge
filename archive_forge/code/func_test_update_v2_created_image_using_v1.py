import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_update_v2_created_image_using_v1(self):
    fields_to_update = {'name': 'bar', 'container_format': 'bare', 'disk_format': 'qcow2'}
    image = self.mgr.update('v2_created_img', **fields_to_update)
    expect_hdrs = {'x-image-meta-name': 'bar', 'x-image-meta-container_format': 'bare', 'x-image-meta-disk_format': 'qcow2', 'x-glance-registry-purge-props': 'false'}
    expect = [('PUT', '/v1/images/v2_created_img', expect_hdrs, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual('v2_created_img', image.id)
    self.assertEqual('bar', image.name)
    self.assertEqual(0, image.size)
    self.assertEqual('bare', image.container_format)
    self.assertEqual('qcow2', image.disk_format)