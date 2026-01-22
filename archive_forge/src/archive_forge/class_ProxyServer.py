import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
class ProxyServer(http_server.HttpServer):
    """A proxy test server for http transports."""
    proxy_requests = True