from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.exc
from glance.api import policy
from glance.common import wsgi
import glance.context
from glance.i18n import _, _LW
class UnauthenticatedContextMiddleware(BaseContextMiddleware):

    def process_request(self, req):
        """Create a context without an authorized user."""
        kwargs = {'user': None, 'tenant': None, 'roles': [], 'is_admin': True}
        req.context = glance.context.RequestContext(**kwargs)