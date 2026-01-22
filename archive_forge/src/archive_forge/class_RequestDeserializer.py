import http.client as http
import glance_store
from oslo_log import log as logging
from oslo_utils import encodeutils
import webob.exc
from glance.api import policy
from glance.api.v2 import images as v2_api
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
import glance.notifier
class RequestDeserializer(wsgi.JSONRequestDeserializer):

    def update(self, request):
        try:
            schema = v2_api.get_schema()
            schema_format = {'tags': [request.urlvars.get('tag_value')]}
            schema.validate(schema_format)
        except exception.InvalidObject as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)
        return super(RequestDeserializer, self).default(request)