import sys
import urllib.parse as urlparse
import glance_store as store_api
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
import glance.db as db_api
from glance.i18n import _LE, _LW
from glance import scrubber
def update_store_in_locations(context, image, image_repo):
    store_updated = False
    for loc in image.locations:
        if not loc['metadata'].get('store') or loc['metadata'].get('store') not in CONF.enabled_backends:
            if loc['url'].startswith('cinder://'):
                _update_cinder_location_and_store_id(context, loc)
            store_id = _get_store_id_from_uri(loc['url'])
            if store_id:
                if 'store' in loc['metadata']:
                    old_store = loc['metadata']['store']
                    if old_store != store_id:
                        LOG.debug("Store '%(old)s' has changed to '%(new)s' by operator, updating the same in the location of image '%(id)s'", {'old': old_store, 'new': store_id, 'id': image.image_id})
                store_updated = True
                loc['metadata']['store'] = store_id
    if store_updated:
        image_repo.save(image)