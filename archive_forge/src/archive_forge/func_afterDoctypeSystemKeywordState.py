from __future__ import absolute_import, division, unicode_literals
from six import unichr as chr
from collections import deque, OrderedDict
from sys import version_info
from .constants import spaceCharacters
from .constants import entities
from .constants import asciiLetters, asciiUpper2Lower
from .constants import digits, hexDigits, EOF
from .constants import tokenTypes, tagTokenTypes
from .constants import replacementCharacters
from ._inputstream import HTMLInputStream
from ._trie import Trie
def afterDoctypeSystemKeywordState(self):
    data = self.stream.char()
    if data in spaceCharacters:
        self.state = self.beforeDoctypeSystemIdentifierState
    elif data in ("'", '"'):
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'unexpected-char-in-doctype'})
        self.stream.unget(data)
        self.state = self.beforeDoctypeSystemIdentifierState
    elif data is EOF:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'eof-in-doctype'})
        self.currentToken['correct'] = False
        self.tokenQueue.append(self.currentToken)
        self.state = self.dataState
    else:
        self.stream.unget(data)
        self.state = self.beforeDoctypeSystemIdentifierState
    return True