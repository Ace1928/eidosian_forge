import os
from ... import urlutils
from . import request
class GetRequest(VfsRequest):

    def do(self, relpath):
        relpath = self.translate_client_path(relpath)
        backing_bytes = self._backing_transport.get_bytes(relpath)
        return request.SuccessfulSmartServerResponse((b'ok',), backing_bytes)