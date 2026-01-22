import copy
import functools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db import utils as db_utils
from glance.i18n import _, _LI, _LW
def _image_format(image_id, **values):
    dt = timeutils.utcnow()
    image = {'id': image_id, 'name': None, 'owner': None, 'locations': [], 'status': 'queued', 'protected': False, 'visibility': 'shared', 'container_format': None, 'disk_format': None, 'min_ram': 0, 'min_disk': 0, 'size': None, 'virtual_size': None, 'checksum': None, 'os_hash_algo': None, 'os_hash_value': None, 'tags': [], 'created_at': dt, 'updated_at': dt, 'deleted_at': None, 'deleted': False, 'os_hidden': False}
    locations = values.pop('locations', None)
    if locations is not None:
        image['locations'] = []
        for location in locations:
            location_ref = _image_location_format(image_id, location['url'], location['metadata'], location['status'])
            image['locations'].append(location_ref)
            DATA['locations'].append(location_ref)
    return _image_update(image, values, values.pop('properties', {}))