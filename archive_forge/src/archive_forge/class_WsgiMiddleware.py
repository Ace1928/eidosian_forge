import webob.dec
from osprofiler import _utils as utils
from osprofiler import profiler
class WsgiMiddleware(object):
    """WSGI Middleware that enables tracing for an application."""

    def __init__(self, application, hmac_keys=None, enabled=False, **kwargs):
        """Initialize middleware with api-paste.ini arguments.

        :application: wsgi app
        :hmac_keys: Only trace header that was signed with one of these
                    hmac keys will be processed. This limitation is
                    essential, because it allows to profile OpenStack
                    by only those who knows this key which helps
                    avoid DDOS.
        :enabled: This middleware can be turned off fully if enabled is False.
        :kwargs: Other keyword arguments.
                 NOTE(tovin07): Currently, this `kwargs` is not used at all.
                 It's here to avoid some extra keyword arguments in local_conf
                 that cause `__init__() got an unexpected keyword argument`.
        """
        self.application = application
        self.name = 'wsgi'
        self.enabled = enabled
        self.hmac_keys = utils.split(hmac_keys or '')

    @classmethod
    def factory(cls, global_conf, **local_conf):

        def filter_(app):
            return cls(app, **local_conf)
        return filter_

    def _trace_is_valid(self, trace_info):
        if not isinstance(trace_info, dict):
            return False
        trace_keys = set(trace_info.keys())
        if not all((k in trace_keys for k in _REQUIRED_KEYS)):
            return False
        if trace_keys.difference(_REQUIRED_KEYS + _OPTIONAL_KEYS):
            return False
        return True

    @webob.dec.wsgify
    def __call__(self, request):
        if _ENABLED is not None and (not _ENABLED) or (_ENABLED is None and (not self.enabled)):
            return request.get_response(self.application)
        trace_info = utils.signed_unpack(request.headers.get(X_TRACE_INFO), request.headers.get(X_TRACE_HMAC), _HMAC_KEYS or self.hmac_keys)
        if not self._trace_is_valid(trace_info):
            return request.get_response(self.application)
        profiler.init(**trace_info)
        info = {'request': {'path': request.path, 'query': request.query_string, 'method': request.method, 'scheme': request.scheme}}
        try:
            with profiler.Trace(self.name, info=info):
                return request.get_response(self.application)
        finally:
            profiler.clean()