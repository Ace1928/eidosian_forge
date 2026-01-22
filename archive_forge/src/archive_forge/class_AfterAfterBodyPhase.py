from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class AfterAfterBodyPhase(Phase):
    __slots__ = tuple()

    def processEOF(self):
        pass

    def processComment(self, token):
        self.tree.insertComment(token, self.tree.document)

    def processSpaceCharacters(self, token):
        return self.parser.phases['inBody'].processSpaceCharacters(token)

    def processCharacters(self, token):
        self.parser.parseError('expected-eof-but-got-char')
        self.parser.phase = self.parser.phases['inBody']
        return token

    def startTagHtml(self, token):
        return self.parser.phases['inBody'].processStartTag(token)

    def startTagOther(self, token):
        self.parser.parseError('expected-eof-but-got-start-tag', {'name': token['name']})
        self.parser.phase = self.parser.phases['inBody']
        return token

    def processEndTag(self, token):
        self.parser.parseError('expected-eof-but-got-end-tag', {'name': token['name']})
        self.parser.phase = self.parser.phases['inBody']
        return token
    startTagHandler = _utils.MethodDispatcher([('html', startTagHtml)])
    startTagHandler.default = startTagOther