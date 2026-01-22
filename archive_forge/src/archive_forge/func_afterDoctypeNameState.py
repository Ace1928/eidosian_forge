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
def afterDoctypeNameState(self):
    data = self.stream.char()
    if data in spaceCharacters:
        pass
    elif data == '>':
        self.tokenQueue.append(self.currentToken)
        self.state = self.dataState
    elif data is EOF:
        self.currentToken['correct'] = False
        self.stream.unget(data)
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'eof-in-doctype'})
        self.tokenQueue.append(self.currentToken)
        self.state = self.dataState
    else:
        if data in ('p', 'P'):
            matched = True
            for expected in (('u', 'U'), ('b', 'B'), ('l', 'L'), ('i', 'I'), ('c', 'C')):
                data = self.stream.char()
                if data not in expected:
                    matched = False
                    break
            if matched:
                self.state = self.afterDoctypePublicKeywordState
                return True
        elif data in ('s', 'S'):
            matched = True
            for expected in (('y', 'Y'), ('s', 'S'), ('t', 'T'), ('e', 'E'), ('m', 'M')):
                data = self.stream.char()
                if data not in expected:
                    matched = False
                    break
            if matched:
                self.state = self.afterDoctypeSystemKeywordState
                return True
        self.stream.unget(data)
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-space-or-right-bracket-in-doctype', 'datavars': {'data': data}})
        self.currentToken['correct'] = False
        self.state = self.bogusDoctypeState
    return True