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
def _normalize_locations(context, image, force_show_deleted=False):
    """
    Generate suitable dictionary list for locations field of image.

    We don't need to set other data fields of location record which return
    from image query.
    """
    if image['status'] == 'deactivated' and (not context.is_admin):
        image['locations'] = []
        return image
    if force_show_deleted:
        locations = image['locations']
    else:
        locations = [x for x in image['locations'] if not x['deleted']]
    image['locations'] = [{'id': loc['id'], 'url': loc['url'], 'metadata': loc['metadata'], 'status': loc['status']} for loc in locations]
    return image