import http.client as http
import re
from oslo_log import log as logging
import webob
from glance.api.common import size_checked_iter
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
import glance.db
from glance.i18n import _LE, _LI
from glance import image_cache
from glance import notifier
def _process_DELETE_response(self, resp, image_id, version=None):
    if self.cache.is_cached(image_id):
        LOG.debug('Removing image %s from cache', image_id)
        self.cache.delete_cached_image(image_id)
    return resp