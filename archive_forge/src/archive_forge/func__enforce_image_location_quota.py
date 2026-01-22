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
def _enforce_image_location_quota(image, locations, is_setter=False):
    if CONF.image_location_quota < 0:
        return
    attempted = len(image.locations) + len(locations)
    attempted = attempted if not is_setter else len(locations)
    maximum = CONF.image_location_quota
    if attempted > maximum:
        raise exception.ImageLocationLimitExceeded(attempted=attempted, maximum=maximum)