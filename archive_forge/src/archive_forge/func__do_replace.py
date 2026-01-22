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
def _do_replace(self, req, image, api_pol, change):
    path = change['path']
    path_root = path[0]
    value = change['value']
    if path_root == 'locations' and (not value):
        msg = _('Cannot set locations to empty list.')
        raise webob.exc.HTTPForbidden(msg)
    elif path_root == 'locations' and value:
        api_pol.update_locations()
        self._do_replace_locations(image, value)
    elif path_root == 'owner' and req.context.is_admin == False:
        msg = _("Owner can't be updated by non admin.")
        raise webob.exc.HTTPForbidden(msg)
    else:
        api_pol.update_property(path_root, value)
        if hasattr(image, path_root):
            setattr(image, path_root, value)
        elif path_root in image.extra_properties:
            image.extra_properties[path_root] = value
        else:
            msg = _('Property %s does not exist.')
            raise webob.exc.HTTPConflict(msg % path_root)