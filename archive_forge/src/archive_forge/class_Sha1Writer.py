from io import BytesIO
import mmap
import os
import sys
import zlib
from gitdb.fun import (
from gitdb.util import (
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_bytes
class Sha1Writer:
    """Simple stream writer which produces a sha whenever you like as it degests
    everything it is supposed to write"""
    __slots__ = 'sha1'

    def __init__(self):
        self.sha1 = make_sha()

    def write(self, data):
        """:raise IOError: If not all bytes could be written
        :param data: byte object
        :return: length of incoming data"""
        self.sha1.update(data)
        return len(data)

    def sha(self, as_hex=False):
        """:return: sha so far
        :param as_hex: if True, sha will be hex-encoded, binary otherwise"""
        if as_hex:
            return self.sha1.hexdigest()
        return self.sha1.digest()