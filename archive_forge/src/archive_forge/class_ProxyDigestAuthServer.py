import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
class ProxyDigestAuthServer(DigestAuthServer, ProxyAuthServer):
    """A proxy server requiring basic authentication"""

    def __init__(self, protocol_version=None):
        ProxyAuthServer.__init__(self, DigestAuthRequestHandler, 'digest', protocol_version=protocol_version)
        self.init_proxy_auth()