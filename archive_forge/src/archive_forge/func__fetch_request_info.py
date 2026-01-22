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
@staticmethod
def _fetch_request_info(request):
    """
        Preserve the cached image id, version for consumption by the
        process_response method of this middleware
        """
    try:
        image_id = request.environ['api.cache.image_id']
        method = request.environ['api.cache.method']
        version = request.environ['api.cache.version']
    except KeyError:
        return None
    else:
        return (image_id, method, version)