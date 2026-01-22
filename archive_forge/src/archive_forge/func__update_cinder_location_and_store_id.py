import sys
import urllib.parse as urlparse
import glance_store as store_api
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
import glance.db as db_api
from glance.i18n import _LE, _LW
from glance import scrubber
def _update_cinder_location_and_store_id(context, loc):
    """Update store location of legacy images

    While upgrading from single cinder store to multiple stores,
    the images having a store configured with a volume type matching
    the image-volume's type will be migrated/associated to that store
    and their location url will be updated respectively to the new format
    i.e. cinder://store-id/volume-id
    If there is no store configured for the image, the location url will
    not be updated.
    """
    uri = loc['url']
    volume_id = loc['url'].split('/')[-1]
    scheme = urlparse.urlparse(uri).scheme
    location_map = store_api.location.SCHEME_TO_CLS_BACKEND_MAP
    if scheme not in location_map:
        LOG.warning(_LW("Unknown scheme '%(scheme)s' found in uri '%(uri)s'"), {'scheme': scheme, 'uri': uri})
        return
    for store in location_map[scheme]:
        store_instance = location_map[scheme][store]['store']
        if store_instance.is_image_associated_with_store(context, volume_id):
            url_prefix = store_instance.url_prefix
            loc['url'] = '%s/%s' % (url_prefix, volume_id)
            loc['metadata']['store'] = '%s' % store
            return
    LOG.warning(_LW("Not able to update location url '%s' of legacy image due to unknown issues."), uri)