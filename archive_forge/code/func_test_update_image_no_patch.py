import io
import operator
import tempfile
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack import connection
from openstack import exceptions
from openstack.image.v1 import image as image_v1
from openstack.image.v2 import image
from openstack.tests import fakes
from openstack.tests.unit import base
def test_update_image_no_patch(self):
    self.cloud.image_api_use_tasks = False
    args = {'name': self.image_name, 'container_format': 'bare', 'disk_format': 'qcow2', 'owner_specified.openstack.md5': fakes.NO_MD5, 'owner_specified.openstack.sha256': fakes.NO_SHA256, 'owner_specified.openstack.object': 'images/{name}'.format(name=self.image_name), 'visibility': 'private'}
    ret = args.copy()
    ret['id'] = self.image_id
    ret['status'] = 'success'
    self.cloud.update_image_properties(image=image.Image.existing(**ret), **{'owner_specified.openstack.object': 'images/{name}'.format(name=self.image_name)})
    self.assert_calls()