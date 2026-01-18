from openstack.cloud import _utils
from openstack import exceptions
from openstack.image.v2._proxy import Proxy
from openstack import utils
def wait_for_image(self, image, timeout=3600):
    image_id = image['id']
    for count in utils.iterate_timeout(timeout, 'Timeout waiting for image to snapshot'):
        image = self.get_image(image_id)
        if not image:
            continue
        if image['status'] == 'active':
            return image
        elif image['status'] == 'error':
            raise exceptions.SDKException('Image {image} hit error state'.format(image=image_id))