import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import image_tags
def test_update_image_tag(self):
    image_id = IMAGE
    tag_value = TAG
    self.controller.update(image_id, tag_value)
    expect = [('PUT', '/v2/images/{image}/tags/{tag_value}'.format(image=IMAGE, tag_value=TAG), {}, None)]
    self.assertEqual(expect, self.api.calls)