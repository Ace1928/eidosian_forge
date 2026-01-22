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
class BaseTestImage(base.TestCase):

    def setUp(self):
        super(BaseTestImage, self).setUp()
        self.image_id = str(uuid.uuid4())
        self.image_name = self.getUniqueString('image')
        self.object_name = 'images/{name}'.format(name=self.image_name)
        self.imagefile = tempfile.NamedTemporaryFile(delete=False)
        data = b'\x02\x00'
        self.imagefile.write(data)
        self.imagefile.close()
        self.output = data
        self.fake_image_dict = fakes.make_fake_image(image_id=self.image_id, image_name=self.image_name, data=self.imagefile.name)
        self.fake_search_return = {'images': [self.fake_image_dict]}
        self.container_name = self.getUniqueString('container')

    def _compare_images(self, exp, real):
        self.assertDictEqual(image.Image(**exp).to_dict(computed=False), real.to_dict(computed=False))

    def _compare_images_v1(self, exp, real):
        self.assertDictEqual(image_v1.Image(**exp).to_dict(computed=False), real.to_dict(computed=False))