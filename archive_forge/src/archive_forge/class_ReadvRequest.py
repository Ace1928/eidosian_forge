import os
from ... import urlutils
from . import request
class ReadvRequest(VfsRequest):

    def do(self, relpath):
        relpath = self.translate_client_path(relpath)
        self._relpath = relpath

    def do_body(self, body_bytes):
        """accept offsets for a readv request."""
        offsets = self._deserialise_offsets(body_bytes)
        backing_bytes = b''.join((bytes for offset, bytes in self._backing_transport.readv(self._relpath, offsets)))
        return request.SuccessfulSmartServerResponse((b'readv',), backing_bytes)

    def _deserialise_offsets(self, text):
        offsets = []
        for line in text.split(b'\n'):
            if not line:
                continue
            start, length = line.split(b',')
            offsets.append((int(start), int(length)))
        return offsets