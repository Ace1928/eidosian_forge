from io import BytesIO
from os import (
import socket
import dulwich
from dulwich.errors import (
class ProtocolFile(object):

    def __init__(self, proto):
        self._proto = proto
        self._offset = 0

    def write(self, data):
        self._proto.write(data)
        self._offset += len(data)

    def tell(self):
        return self._offset

    def close(self):
        pass