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
class ContentAttrParser(object):

    def __init__(self, data):
        assert isinstance(data, bytes)
        self.data = data

    def parse(self):
        try:
            self.data.jumpTo(b'charset')
            self.data.position += 1
            self.data.skip()
            if not self.data.currentByte == b'=':
                return None
            self.data.position += 1
            self.data.skip()
            if self.data.currentByte in (b'"', b"'"):
                quoteMark = self.data.currentByte
                self.data.position += 1
                oldPosition = self.data.position
                if self.data.jumpTo(quoteMark):
                    return self.data[oldPosition:self.data.position]
                else:
                    return None
            else:
                oldPosition = self.data.position
                try:
                    self.data.skipUntil(spaceCharactersBytes)
                    return self.data[oldPosition:self.data.position]
                except StopIteration:
                    return self.data[oldPosition:]
        except StopIteration:
            return None