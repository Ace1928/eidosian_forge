import http.client as http
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import webob.exc
from wsme.rest import json
from glance.api import policy
from glance.api.v2 import metadef_namespaces as namespaces
import glance.api.v2.metadef_properties as properties
from glance.api.v2.model.metadef_object import MetadefObject
from glance.api.v2.model.metadef_object import MetadefObjects
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
from glance.common import wsme_utils
import glance.db
from glance.i18n import _
import glance.notifier
import glance.schema
Metadef objects resource factory method