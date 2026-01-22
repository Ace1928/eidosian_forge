import datetime
import hashlib
import http.client as http
import os
import re
import urllib.parse as urlparse
import uuid
from castellan.common import exception as castellan_exception
from castellan import key_manager
import glance_store
from glance_store import location
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from oslo_utils import encodeutils
from oslo_utils import timeutils as oslo_timeutils
import requests
import webob.exc
from glance.api import common
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
from glance import context as glance_context
import glance.db
import glance.gateway
from glance.i18n import _, _LE, _LI, _LW
import glance.notifier
from glance.quota import keystone as ks_quota
import glance.schema
def _do_remove_locations(self, image, path_pos):
    if CONF.show_multiple_locations == False:
        msg = _("It's not allowed to remove locations if locations are invisible.")
        raise webob.exc.HTTPForbidden(explanation=msg)
    if image.status not in 'active':
        msg = _("It's not allowed to remove locations if image status is %s.") % image.status
        raise webob.exc.HTTPConflict(explanation=msg)
    if len(image.locations) == 1:
        LOG.debug('User forbidden to remove last location of image %s', image.image_id)
        msg = _('Cannot remove last location in the image.')
        raise exception.Forbidden(msg)
    pos = self._get_locations_op_pos(path_pos, len(image.locations), False)
    if pos is None:
        msg = _('Invalid position for removing a location.')
        raise webob.exc.HTTPBadRequest(explanation=msg)
    try:
        image.locations.pop(pos)
    except Exception as e:
        raise webob.exc.HTTPInternalServerError(explanation=encodeutils.exception_to_unicode(e))