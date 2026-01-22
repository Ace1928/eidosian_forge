from oslo_config import cfg
from oslo_middleware import base
class HTTPProxyToWSGI(base.ConfigurableMiddleware):
    """HTTP proxy to WSGI termination middleware.

    This middleware overloads WSGI environment variables with the one provided
    by the remote HTTP reverse proxy.

    """

    def __init__(self, application, *args, **kwargs):
        super(HTTPProxyToWSGI, self).__init__(application, *args, **kwargs)
        self.oslo_conf.register_opts(OPTS, group='oslo_middleware')

    @staticmethod
    def _parse_rfc7239_header(header):
        """Parses RFC7239 Forward headers.

        e.g. for=192.0.2.60;proto=http, for=192.0.2.60;by=203.0.113.43

        """
        result = []
        for proxy in header.split(','):
            entry = {}
            for d in proxy.split(';'):
                key, _, value = d.partition('=')
                entry[key.lower().strip()] = value.strip()
            result.append(entry)
        return result

    def process_request(self, req):
        if not self._conf_get('enable_proxy_headers_parsing'):
            return
        fwd_hdr = req.environ.get('HTTP_FORWARDED')
        if fwd_hdr:
            proxies = self._parse_rfc7239_header(fwd_hdr)
            if proxies:
                proxy = proxies[0]
                forwarded_proto = proxy.get('proto')
                if forwarded_proto:
                    req.environ['wsgi.url_scheme'] = forwarded_proto
                forwarded_host = proxy.get('host')
                if forwarded_host:
                    req.environ['HTTP_HOST'] = forwarded_host
                forwarded_for = proxy.get('for')
                if forwarded_for:
                    req.environ['REMOTE_ADDR'] = forwarded_for
        else:
            forwarded_proto = req.environ.get('HTTP_X_FORWARDED_PROTO')
            if forwarded_proto:
                req.environ['wsgi.url_scheme'] = forwarded_proto
            forwarded_host = req.environ.get('HTTP_X_FORWARDED_HOST')
            if forwarded_host:
                req.environ['HTTP_HOST'] = forwarded_host
            forwarded_for = req.environ.get('HTTP_X_FORWARDED_FOR')
            if forwarded_for:
                req.environ['REMOTE_ADDR'] = forwarded_for
        v = req.environ.get('HTTP_X_FORWARDED_PREFIX')
        if v:
            req.environ['SCRIPT_NAME'] = v + req.environ['SCRIPT_NAME']