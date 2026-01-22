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
def _get_change_operation_d4(self, raw_change):
    op = None
    for key in self._supported_operations:
        if key in raw_change:
            if op is not None:
                msg = _('Operation objects must contain only one member named "add", "remove", or "replace".')
                raise webob.exc.HTTPBadRequest(explanation=msg)
            op = key
    if op is None:
        msg = _('Operation objects must contain exactly one member named "add", "remove", or "replace".')
        raise webob.exc.HTTPBadRequest(explanation=msg)
    return op