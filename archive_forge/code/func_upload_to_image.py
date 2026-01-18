from openstack.common import metadata
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack import utils
def upload_to_image(self, session, image_name, force=False, disk_format=None, container_format=None, visibility=None, protected=None):
    """Upload the volume to image service"""
    req = dict(image_name=image_name, force=force)
    if disk_format is not None:
        req['disk_format'] = disk_format
    if container_format is not None:
        req['container_format'] = container_format
    if visibility is not None:
        req['visibility'] = visibility
    if protected is not None:
        req['protected'] = protected
    if visibility is not None or protected is not None:
        utils.require_microversion(session, '3.1')
    body = {'os-volume_upload_image': req}
    resp = self._action(session, body).json()
    return resp['os-volume_upload_image']