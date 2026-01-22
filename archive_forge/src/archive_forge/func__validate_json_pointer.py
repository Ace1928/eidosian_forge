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
def _validate_json_pointer(self, pointer):
    """Validate a json pointer.

        We only accept a limited form of json pointers.
        """
    if not pointer.startswith('/'):
        msg = _('Pointer `%s` does not start with "/".') % pointer
        raise webob.exc.HTTPBadRequest(explanation=msg)
    if re.search('/\\s*?/', pointer[1:]):
        msg = _('Pointer `%s` contains adjacent "/".') % pointer
        raise webob.exc.HTTPBadRequest(explanation=msg)
    if len(pointer) > 1 and pointer.endswith('/'):
        msg = _('Pointer `%s` end with "/".') % pointer
        raise webob.exc.HTTPBadRequest(explanation=msg)
    if pointer[1:].strip() == '/':
        msg = _('Pointer `%s` does not contains valid token.') % pointer
        raise webob.exc.HTTPBadRequest(explanation=msg)
    if re.search('~[^01]', pointer) or pointer.endswith('~'):
        msg = _('Pointer `%s` contains "~" not part of a recognized escape sequence.') % pointer
        raise webob.exc.HTTPBadRequest(explanation=msg)