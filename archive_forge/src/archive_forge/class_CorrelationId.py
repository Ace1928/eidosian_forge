from oslo_utils import uuidutils
from oslo_middleware import base
class CorrelationId(base.ConfigurableMiddleware):
    """Middleware that attaches a correlation id to WSGI request"""

    def process_request(self, req):
        correlation_id = req.headers.get('X_CORRELATION_ID') or uuidutils.generate_uuid()
        req.headers['X_CORRELATION_ID'] = correlation_id