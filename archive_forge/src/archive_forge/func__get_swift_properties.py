import copy
import glance_store as g_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_serialization.jsonutils as json
import webob.exc
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
import glance.db
from glance.i18n import _
from glance.quota import keystone as ks_quota
@staticmethod
def _get_swift_properties(store_detail):
    return {'container': store_detail.container, 'large_object_size': store_detail.large_object_size, 'large_object_chunk_size': store_detail.large_object_chunk_size}