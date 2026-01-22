import os
from ... import urlutils
from . import request
class HasRequest(VfsRequest):

    def do(self, relpath):
        relpath = self.translate_client_path(relpath)
        r = self._backing_transport.has(relpath) and b'yes' or b'no'
        return request.SuccessfulSmartServerResponse((r,))