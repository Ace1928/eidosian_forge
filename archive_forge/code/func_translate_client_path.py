import os
from ... import urlutils
from . import request
def translate_client_path(self, relpath):
    x = request.SmartServerRequest.translate_client_path(self, relpath)
    return str(urlutils.unescape(x))