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
def _validate_change(self, change):
    path_root = change['path'][0]
    if path_root in self._readonly_properties:
        msg = _("Attribute '%s' is read-only.") % path_root
        raise webob.exc.HTTPForbidden(explanation=msg)
    if path_root in self._reserved_properties:
        msg = _("Attribute '%s' is reserved.") % path_root
        raise webob.exc.HTTPForbidden(explanation=msg)
    if any((path_root.startswith(ns) for ns in self._reserved_namespaces)):
        msg = _("Attribute '%s' is reserved.") % path_root
        raise webob.exc.HTTPForbidden(explanation=msg)
    if change['op'] == 'remove':
        return
    partial_image = None
    if len(change['path']) == 1:
        partial_image = {path_root: change['value']}
    elif path_root in get_base_properties().keys() and get_base_properties()[path_root].get('type', '') == 'array':
        partial_image = {path_root: [change['value']]}
    if partial_image:
        try:
            self.schema.validate(partial_image)
        except exception.InvalidObject as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)