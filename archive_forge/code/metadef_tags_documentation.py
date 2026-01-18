import http.client as http
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
import webob.exc
from wsme.rest import json
from glance.api import policy
from glance.api.v2.model.metadef_tag import MetadefTag
from glance.api.v2.model.metadef_tag import MetadefTags
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
from glance.common import wsme_utils
import glance.db
from glance.i18n import _
import glance.notifier
import glance.schema
Metadef tags resource factory method