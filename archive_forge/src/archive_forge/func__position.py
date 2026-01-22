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
def _position(self, offset):
    chunk = self.chunk
    nLines = chunk.count('\n', 0, offset)
    positionLine = self.prevNumLines + nLines
    lastLinePos = chunk.rfind('\n', 0, offset)
    if lastLinePos == -1:
        positionColumn = self.prevNumCols + offset
    else:
        positionColumn = offset - (lastLinePos + 1)
    return (positionLine, positionColumn)