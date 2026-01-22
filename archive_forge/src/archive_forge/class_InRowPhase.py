from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InRowPhase(Phase):
    __slots__ = tuple()

    def clearStackToTableRowContext(self):
        while self.tree.openElements[-1].name not in ('tr', 'html'):
            self.parser.parseError('unexpected-implied-end-tag-in-table-row', {'name': self.tree.openElements[-1].name})
            self.tree.openElements.pop()

    def ignoreEndTagTr(self):
        return not self.tree.elementInScope('tr', variant='table')

    def processEOF(self):
        self.parser.phases['inTable'].processEOF()

    def processSpaceCharacters(self, token):
        return self.parser.phases['inTable'].processSpaceCharacters(token)

    def processCharacters(self, token):
        return self.parser.phases['inTable'].processCharacters(token)

    def startTagTableCell(self, token):
        self.clearStackToTableRowContext()
        self.tree.insertElement(token)
        self.parser.phase = self.parser.phases['inCell']
        self.tree.activeFormattingElements.append(Marker)

    def startTagTableOther(self, token):
        ignoreEndTag = self.ignoreEndTagTr()
        self.endTagTr(impliedTagToken('tr'))
        if not ignoreEndTag:
            return token

    def startTagOther(self, token):
        return self.parser.phases['inTable'].processStartTag(token)

    def endTagTr(self, token):
        if not self.ignoreEndTagTr():
            self.clearStackToTableRowContext()
            self.tree.openElements.pop()
            self.parser.phase = self.parser.phases['inTableBody']
        else:
            assert self.parser.innerHTML
            self.parser.parseError()

    def endTagTable(self, token):
        ignoreEndTag = self.ignoreEndTagTr()
        self.endTagTr(impliedTagToken('tr'))
        if not ignoreEndTag:
            return token

    def endTagTableRowGroup(self, token):
        if self.tree.elementInScope(token['name'], variant='table'):
            self.endTagTr(impliedTagToken('tr'))
            return token
        else:
            self.parser.parseError()

    def endTagIgnore(self, token):
        self.parser.parseError('unexpected-end-tag-in-table-row', {'name': token['name']})

    def endTagOther(self, token):
        return self.parser.phases['inTable'].processEndTag(token)
    startTagHandler = _utils.MethodDispatcher([('html', Phase.startTagHtml), (('td', 'th'), startTagTableCell), (('caption', 'col', 'colgroup', 'tbody', 'tfoot', 'thead', 'tr'), startTagTableOther)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([('tr', endTagTr), ('table', endTagTable), (('tbody', 'tfoot', 'thead'), endTagTableRowGroup), (('body', 'caption', 'col', 'colgroup', 'html', 'td', 'th'), endTagIgnore)])
    endTagHandler.default = endTagOther