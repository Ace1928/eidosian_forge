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
def _enforce_image_property_quota(self, properties):
    if CONF.image_property_quota < 0:
        return
    attempted = len([x for x in properties.keys() if not x.startswith(glance.api.common.GLANCE_RESERVED_NS)])
    maximum = CONF.image_property_quota
    if attempted > maximum:
        kwargs = {'attempted': attempted, 'maximum': maximum}
        exc = exception.ImagePropertyLimitExceeded(**kwargs)
        LOG.debug(encodeutils.exception_to_unicode(exc))
        raise exc