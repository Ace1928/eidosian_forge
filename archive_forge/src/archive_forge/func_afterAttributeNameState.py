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
def afterAttributeNameState(self):
    data = self.stream.char()
    if data in spaceCharacters:
        self.stream.charsUntil(spaceCharacters, True)
    elif data == '=':
        self.state = self.beforeAttributeValueState
    elif data == '>':
        self.emitCurrentToken()
    elif data in asciiLetters:
        self.currentToken['data'].append([data, ''])
        self.state = self.attributeNameState
    elif data == '/':
        self.state = self.selfClosingStartTagState
    elif data == '\x00':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        self.currentToken['data'].append(['�', ''])
        self.state = self.attributeNameState
    elif data in ("'", '"', '<'):
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-character-after-attribute-name'})
        self.currentToken['data'].append([data, ''])
        self.state = self.attributeNameState
    elif data is EOF:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-end-of-tag-but-got-eof'})
        self.state = self.dataState
    else:
        self.currentToken['data'].append([data, ''])
        self.state = self.attributeNameState
    return True