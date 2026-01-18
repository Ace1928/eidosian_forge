from __future__ import print_function, absolute_import
import io
from .common import DTDForbidden, EntitiesForbidden, ExternalReferenceForbidden, PY3
def unmonkey_patch():
    xmlrpc_client.FastParser = None
    xmlrpc_client.GzipDecodedResponse = _OrigGzipDecodedResponse
    xmlrpc_client.gzip_decode = _orig_gzip_decode
    if xmlrpc_server:
        xmlrpc_server.gzip_decode = _orig_gzip_decode