import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
class BaseChecksum:
    _CHUNK_SIZE = 1024 * 1024

    def update(self, chunk):
        pass

    def digest(self):
        pass

    def b64digest(self):
        bs = self.digest()
        return base64.b64encode(bs).decode('ascii')

    def _handle_fileobj(self, fileobj):
        start_position = fileobj.tell()
        for chunk in iter(lambda: fileobj.read(self._CHUNK_SIZE), b''):
            self.update(chunk)
        fileobj.seek(start_position)

    def handle(self, body):
        if isinstance(body, (bytes, bytearray)):
            self.update(body)
        else:
            self._handle_fileobj(body)
        return self.b64digest()