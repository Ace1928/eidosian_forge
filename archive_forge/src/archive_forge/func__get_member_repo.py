import copy
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import webob
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
import glance.notifier
import glance.schema
def _get_member_repo(self, req, image):
    try:
        return self.gateway.get_member_repo(image, req.context)
    except exception.Forbidden as e:
        msg = _('Error fetching members of image %(image_id)s: %(inner_msg)s') % {'image_id': image.image_id, 'inner_msg': e.msg}
        LOG.warning(msg)
        raise webob.exc.HTTPForbidden(explanation=msg)