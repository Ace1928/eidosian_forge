import copy
import glance_store as store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
import glance.api.common
import glance.common.exception as exception
from glance.common import utils
import glance.domain
import glance.domain.proxy
from glance.i18n import _, _LI
def _calc_required_size(context, image, locations):
    required_size = None
    if image.size:
        required_size = image.size * len(locations)
    else:
        for location in locations:
            size_from_backend = None
            try:
                if CONF.enabled_backends:
                    size_from_backend = store.get_size_from_uri_and_backend(location['url'], location['metadata'].get('store'), context=context)
                else:
                    size_from_backend = store.get_size_from_backend(location['url'], context=context)
            except (store.UnknownScheme, store.NotFound):
                pass
            except store.BadStoreUri:
                raise exception.BadStoreUri
            if size_from_backend:
                required_size = size_from_backend * len(locations)
                break
    return required_size