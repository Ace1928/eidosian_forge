import errno
import hashlib
import testtools
from unittest import mock
import ddt
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import images
def test_update_add_prop(self):
    image_id = '3a4560a1-e585-443e-9b39-553b46ec92d1'
    params = {'finn': 'human'}
    image = self.controller.update(image_id, **params)
    expect_hdrs = {'Content-Type': 'application/openstack-images-v2.1-json-patch'}
    expect_body = [[('op', 'add'), ('path', '/finn'), ('value', 'human')]]
    expect = [('GET', '/v2/images/%s' % image_id, {}, None), ('PATCH', '/v2/images/%s' % image_id, expect_hdrs, expect_body), ('GET', '/v2/images/%s' % image_id, {'x-openstack-request-id': 'req-1234'}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(image_id, image.id)
    self.assertEqual('image-1', image.name)