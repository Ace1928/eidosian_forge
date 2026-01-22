import sys
import threading
import traceback
import warnings
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_imports import xmlrpclib, _queue
from _pydevd_bundle.pydevd_constants import Null
def _encode_if_needed(obj):
    if isinstance(obj, str):
        return xmlrpclib.Binary(obj.encode('ISO-8859-1', 'xmlcharrefreplace'))
    elif isinstance(obj, bytes):
        try:
            return xmlrpclib.Binary(obj.decode(sys.stdin.encoding).encode('ISO-8859-1', 'xmlcharrefreplace'))
        except:
            return xmlrpclib.Binary(obj)
    return obj