from __future__ import absolute_import, division, unicode_literals
from six import text_type
from six.moves import http_client, urllib
import codecs
import re
from io import BytesIO, StringIO
from tensorboard._vendor import webencodings
from .constants import EOF, spaceCharacters, asciiLetters, asciiUppercase
from .constants import _ReparseException
from . import _utils
def _readFromBuffer(self, bytes):
    remainingBytes = bytes
    rv = []
    bufferIndex = self.position[0]
    bufferOffset = self.position[1]
    while bufferIndex < len(self.buffer) and remainingBytes != 0:
        assert remainingBytes > 0
        bufferedData = self.buffer[bufferIndex]
        if remainingBytes <= len(bufferedData) - bufferOffset:
            bytesToRead = remainingBytes
            self.position = [bufferIndex, bufferOffset + bytesToRead]
        else:
            bytesToRead = len(bufferedData) - bufferOffset
            self.position = [bufferIndex, len(bufferedData)]
            bufferIndex += 1
        rv.append(bufferedData[bufferOffset:bufferOffset + bytesToRead])
        remainingBytes -= bytesToRead
        bufferOffset = 0
    if remainingBytes:
        rv.append(self._readStream(remainingBytes))
    return b''.join(rv)