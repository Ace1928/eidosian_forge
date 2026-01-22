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
class EncodingParser(object):
    """Mini parser for detecting character encoding from meta elements"""

    def __init__(self, data):
        """string - the data to work on for encoding detection"""
        self.data = EncodingBytes(data)
        self.encoding = None

    def getEncoding(self):
        if b'<meta' not in self.data:
            return None
        methodDispatch = ((b'<!--', self.handleComment), (b'<meta', self.handleMeta), (b'</', self.handlePossibleEndTag), (b'<!', self.handleOther), (b'<?', self.handleOther), (b'<', self.handlePossibleStartTag))
        for _ in self.data:
            keepParsing = True
            try:
                self.data.jumpTo(b'<')
            except StopIteration:
                break
            for key, method in methodDispatch:
                if self.data.matchBytes(key):
                    try:
                        keepParsing = method()
                        break
                    except StopIteration:
                        keepParsing = False
                        break
            if not keepParsing:
                break
        return self.encoding

    def handleComment(self):
        """Skip over comments"""
        return self.data.jumpTo(b'-->')

    def handleMeta(self):
        if self.data.currentByte not in spaceCharactersBytes:
            return True
        hasPragma = False
        pendingEncoding = None
        while True:
            attr = self.getAttribute()
            if attr is None:
                return True
            elif attr[0] == b'http-equiv':
                hasPragma = attr[1] == b'content-type'
                if hasPragma and pendingEncoding is not None:
                    self.encoding = pendingEncoding
                    return False
            elif attr[0] == b'charset':
                tentativeEncoding = attr[1]
                codec = lookupEncoding(tentativeEncoding)
                if codec is not None:
                    self.encoding = codec
                    return False
            elif attr[0] == b'content':
                contentParser = ContentAttrParser(EncodingBytes(attr[1]))
                tentativeEncoding = contentParser.parse()
                if tentativeEncoding is not None:
                    codec = lookupEncoding(tentativeEncoding)
                    if codec is not None:
                        if hasPragma:
                            self.encoding = codec
                            return False
                        else:
                            pendingEncoding = codec

    def handlePossibleStartTag(self):
        return self.handlePossibleTag(False)

    def handlePossibleEndTag(self):
        next(self.data)
        return self.handlePossibleTag(True)

    def handlePossibleTag(self, endTag):
        data = self.data
        if data.currentByte not in asciiLettersBytes:
            if endTag:
                data.previous()
                self.handleOther()
            return True
        c = data.skipUntil(spacesAngleBrackets)
        if c == b'<':
            data.previous()
        else:
            attr = self.getAttribute()
            while attr is not None:
                attr = self.getAttribute()
        return True

    def handleOther(self):
        return self.data.jumpTo(b'>')

    def getAttribute(self):
        """Return a name,value pair for the next attribute in the stream,
        if one is found, or None"""
        data = self.data
        c = data.skip(spaceCharactersBytes | frozenset([b'/']))
        assert c is None or len(c) == 1
        if c in (b'>', None):
            return None
        attrName = []
        attrValue = []
        while True:
            if c == b'=' and attrName:
                break
            elif c in spaceCharactersBytes:
                c = data.skip()
                break
            elif c in (b'/', b'>'):
                return (b''.join(attrName), b'')
            elif c in asciiUppercaseBytes:
                attrName.append(c.lower())
            elif c is None:
                return None
            else:
                attrName.append(c)
            c = next(data)
        if c != b'=':
            data.previous()
            return (b''.join(attrName), b'')
        next(data)
        c = data.skip()
        if c in (b"'", b'"'):
            quoteChar = c
            while True:
                c = next(data)
                if c == quoteChar:
                    next(data)
                    return (b''.join(attrName), b''.join(attrValue))
                elif c in asciiUppercaseBytes:
                    attrValue.append(c.lower())
                else:
                    attrValue.append(c)
        elif c == b'>':
            return (b''.join(attrName), b'')
        elif c in asciiUppercaseBytes:
            attrValue.append(c.lower())
        elif c is None:
            return None
        else:
            attrValue.append(c)
        while True:
            c = next(data)
            if c in spacesAngleBrackets:
                return (b''.join(attrName), b''.join(attrValue))
            elif c in asciiUppercaseBytes:
                attrValue.append(c.lower())
            elif c is None:
                return None
            else:
                attrValue.append(c)