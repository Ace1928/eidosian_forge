import logging
from oslo_config import cfg
import webob.dec
import webob.exc
from oslo_middleware._i18n import _
from oslo_middleware import base
class RequestBodySizeLimiter(base.ConfigurableMiddleware):
    """Limit the size of incoming requests."""

    def __init__(self, application, conf=None):
        super(RequestBodySizeLimiter, self).__init__(application, conf)
        self.oslo_conf.register_opts(_opts, group='oslo_middleware')

    @webob.dec.wsgify
    def __call__(self, req):
        max_size = self._conf_get('max_request_body_size')
        if req.content_length is not None and req.content_length > max_size:
            msg = _('Request is too large. Larger than max_request_body_size (%s).') % max_size
            LOG.info(msg)
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=msg)
        if req.content_length is None:
            limiter = LimitingReader(req.body_file, max_size)
            req.body_file = limiter
        return self.application