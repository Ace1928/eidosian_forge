import ssl
from . import http_server, ssl_certs, test_server
class HTTPSServer_urllib(HTTPSServer):
    """Subclass of HTTPSServer that gives https+urllib urls.

    This is for use in testing: connections to this server will always go
    through urllib where possible.
    """
    _url_protocol = 'https+urllib'