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
def _check_user_storage_quota(self, locations):
    required_size = _calc_required_size(self.context, self.image, locations)
    glance.api.common.check_quota(self.context, required_size, self.db_api)
    _enforce_image_location_quota(self.image, locations)