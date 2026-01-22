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
class HTMLUnicodeInputStream(object):
    """Provides a unicode stream of characters to the HTMLTokenizer.

    This class takes care of character encoding and removing or replacing
    incorrect byte-sequences and also provides column and line tracking.

    """
    _defaultChunkSize = 10240

    def __init__(self, source):
        """Initialises the HTMLInputStream.

        HTMLInputStream(source, [encoding]) -> Normalized stream from source
        for use by html5lib.

        source can be either a file-object, local filename or a string.

        The optional encoding parameter must be a string that indicates
        the encoding.  If specified, that encoding will be used,
        regardless of any BOM or later declaration (such as in a meta
        element)

        """
        if not _utils.supports_lone_surrogates:
            self.reportCharacterErrors = None
        elif len('\U0010ffff') == 1:
            self.reportCharacterErrors = self.characterErrorsUCS4
        else:
            self.reportCharacterErrors = self.characterErrorsUCS2
        self.newLines = [0]
        self.charEncoding = (lookupEncoding('utf-8'), 'certain')
        self.dataStream = self.openStream(source)
        self.reset()

    def reset(self):
        self.chunk = ''
        self.chunkSize = 0
        self.chunkOffset = 0
        self.errors = []
        self.prevNumLines = 0
        self.prevNumCols = 0
        self._bufferedCharacter = None

    def openStream(self, source):
        """Produces a file object from source.

        source can be either a file object, local filename or a string.

        """
        if hasattr(source, 'read'):
            stream = source
        else:
            stream = StringIO(source)
        return stream

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

    def position(self):
        """Returns (line, col) of the current position in the stream."""
        line, col = self._position(self.chunkOffset)
        return (line + 1, col)

    def char(self):
        """ Read one character from the stream or queue if available. Return
            EOF when EOF is reached.
        """
        if self.chunkOffset >= self.chunkSize:
            if not self.readChunk():
                return EOF
        chunkOffset = self.chunkOffset
        char = self.chunk[chunkOffset]
        self.chunkOffset = chunkOffset + 1
        return char

    def readChunk(self, chunkSize=None):
        if chunkSize is None:
            chunkSize = self._defaultChunkSize
        self.prevNumLines, self.prevNumCols = self._position(self.chunkSize)
        self.chunk = ''
        self.chunkSize = 0
        self.chunkOffset = 0
        data = self.dataStream.read(chunkSize)
        if self._bufferedCharacter:
            data = self._bufferedCharacter + data
            self._bufferedCharacter = None
        elif not data:
            return False
        if len(data) > 1:
            lastv = ord(data[-1])
            if lastv == 13 or 55296 <= lastv <= 56319:
                self._bufferedCharacter = data[-1]
                data = data[:-1]
        if self.reportCharacterErrors:
            self.reportCharacterErrors(data)
        data = data.replace('\r\n', '\n')
        data = data.replace('\r', '\n')
        self.chunk = data
        self.chunkSize = len(data)
        return True

    def characterErrorsUCS4(self, data):
        for _ in range(len(invalid_unicode_re.findall(data))):
            self.errors.append('invalid-codepoint')

    def characterErrorsUCS2(self, data):
        skip = False
        for match in invalid_unicode_re.finditer(data):
            if skip:
                continue
            codepoint = ord(match.group())
            pos = match.start()
            if _utils.isSurrogatePair(data[pos:pos + 2]):
                char_val = _utils.surrogatePairToCodepoint(data[pos:pos + 2])
                if char_val in non_bmp_invalid_codepoints:
                    self.errors.append('invalid-codepoint')
                skip = True
            elif codepoint >= 55296 and codepoint <= 57343 and (pos == len(data) - 1):
                self.errors.append('invalid-codepoint')
            else:
                skip = False
                self.errors.append('invalid-codepoint')

    def charsUntil(self, characters, opposite=False):
        """ Returns a string of characters from the stream up to but not
        including any character in 'characters' or EOF. 'characters' must be
        a container that supports the 'in' method and iteration over its
        characters.
        """
        try:
            chars = charsUntilRegEx[characters, opposite]
        except KeyError:
            if __debug__:
                for c in characters:
                    assert ord(c) < 128
            regex = ''.join(['\\x%02x' % ord(c) for c in characters])
            if not opposite:
                regex = '^%s' % regex
            chars = charsUntilRegEx[characters, opposite] = re.compile('[%s]+' % regex)
        rv = []
        while True:
            m = chars.match(self.chunk, self.chunkOffset)
            if m is None:
                if self.chunkOffset != self.chunkSize:
                    break
            else:
                end = m.end()
                if end != self.chunkSize:
                    rv.append(self.chunk[self.chunkOffset:end])
                    self.chunkOffset = end
                    break
            rv.append(self.chunk[self.chunkOffset:])
            if not self.readChunk():
                break
        r = ''.join(rv)
        return r

    def unget(self, char):
        if char is not EOF:
            if self.chunkOffset == 0:
                self.chunk = char + self.chunk
                self.chunkSize += 1
            else:
                self.chunkOffset -= 1
                assert self.chunk[self.chunkOffset] == char