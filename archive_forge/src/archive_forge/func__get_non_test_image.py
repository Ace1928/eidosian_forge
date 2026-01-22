from openstack.tests.functional import base
from openstack.tests.functional.image.v2.test_image import TEST_IMAGE_NAME
def _get_non_test_image(self):
    images = self.conn.compute.images()
    image = next(images)
    if image.name == TEST_IMAGE_NAME:
        image = next(images)
    return image