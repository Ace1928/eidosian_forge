import queue
import threading
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
from glance import image_cache
import glance.notifier
class CachedImageSerializer(wsgi.JSONResponseSerializer):

    def queue_image_from_api(self, response, result):
        response.status_int = 202

    def clear_cache(self, response, result):
        response.status_int = 204

    def delete_cache_entry(self, response, result):
        response.status_int = 204