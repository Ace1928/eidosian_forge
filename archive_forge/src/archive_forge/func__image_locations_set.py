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
def _image_locations_set(context, image_id, locations):
    used_loc_ids = [loc['id'] for loc in locations if loc.get('id')]
    image = DATA['images'][image_id]
    for loc in image['locations']:
        if loc['id'] not in used_loc_ids and (not loc['deleted']):
            image_location_delete(context, image_id, loc['id'], 'deleted')
    for i, loc in enumerate(DATA['locations']):
        if loc['image_id'] == image_id and loc['id'] not in used_loc_ids and (not loc['deleted']):
            del DATA['locations'][i]
    for loc in locations:
        if loc.get('id') is None:
            image_location_add(context, image_id, loc)
        else:
            image_location_update(context, image_id, loc)