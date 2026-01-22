import os
from ... import urlutils
from . import request
class AppendRequest(VfsRequest):

    def do(self, relpath, mode):
        relpath = self.translate_client_path(relpath)
        self._relpath = relpath
        self._mode = _deserialise_optional_mode(mode)

    def do_body(self, body_bytes):
        old_length = self._backing_transport.append_bytes(self._relpath, body_bytes, self._mode)
        return request.SuccessfulSmartServerResponse((b'appended', str(old_length).encode('ascii')))