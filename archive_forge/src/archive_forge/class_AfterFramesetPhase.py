from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class AfterFramesetPhase(Phase):
    __slots__ = tuple()

    def processEOF(self):
        pass

    def processCharacters(self, token):
        self.parser.parseError('unexpected-char-after-frameset')

    def startTagNoframes(self, token):
        return self.parser.phases['inHead'].processStartTag(token)

    def startTagOther(self, token):
        self.parser.parseError('unexpected-start-tag-after-frameset', {'name': token['name']})

    def endTagHtml(self, token):
        self.parser.phase = self.parser.phases['afterAfterFrameset']

    def endTagOther(self, token):
        self.parser.parseError('unexpected-end-tag-after-frameset', {'name': token['name']})
    startTagHandler = _utils.MethodDispatcher([('html', Phase.startTagHtml), ('noframes', startTagNoframes)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([('html', endTagHtml)])
    endTagHandler.default = endTagOther