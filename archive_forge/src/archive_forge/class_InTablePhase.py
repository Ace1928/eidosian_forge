from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InTablePhase(Phase):
    __slots__ = tuple()

    def clearStackToTableContext(self):
        while self.tree.openElements[-1].name not in ('table', 'html'):
            self.tree.openElements.pop()

    def processEOF(self):
        if self.tree.openElements[-1].name != 'html':
            self.parser.parseError('eof-in-table')
        else:
            assert self.parser.innerHTML

    def processSpaceCharacters(self, token):
        originalPhase = self.parser.phase
        self.parser.phase = self.parser.phases['inTableText']
        self.parser.phase.originalPhase = originalPhase
        self.parser.phase.processSpaceCharacters(token)

    def processCharacters(self, token):
        originalPhase = self.parser.phase
        self.parser.phase = self.parser.phases['inTableText']
        self.parser.phase.originalPhase = originalPhase
        self.parser.phase.processCharacters(token)

    def insertText(self, token):
        self.tree.insertFromTable = True
        self.parser.phases['inBody'].processCharacters(token)
        self.tree.insertFromTable = False

    def startTagCaption(self, token):
        self.clearStackToTableContext()
        self.tree.activeFormattingElements.append(Marker)
        self.tree.insertElement(token)
        self.parser.phase = self.parser.phases['inCaption']

    def startTagColgroup(self, token):
        self.clearStackToTableContext()
        self.tree.insertElement(token)
        self.parser.phase = self.parser.phases['inColumnGroup']

    def startTagCol(self, token):
        self.startTagColgroup(impliedTagToken('colgroup', 'StartTag'))
        return token

    def startTagRowGroup(self, token):
        self.clearStackToTableContext()
        self.tree.insertElement(token)
        self.parser.phase = self.parser.phases['inTableBody']

    def startTagImplyTbody(self, token):
        self.startTagRowGroup(impliedTagToken('tbody', 'StartTag'))
        return token

    def startTagTable(self, token):
        self.parser.parseError('unexpected-start-tag-implies-end-tag', {'startName': 'table', 'endName': 'table'})
        self.parser.phase.processEndTag(impliedTagToken('table'))
        if not self.parser.innerHTML:
            return token

    def startTagStyleScript(self, token):
        return self.parser.phases['inHead'].processStartTag(token)

    def startTagInput(self, token):
        if 'type' in token['data'] and token['data']['type'].translate(asciiUpper2Lower) == 'hidden':
            self.parser.parseError('unexpected-hidden-input-in-table')
            self.tree.insertElement(token)
            self.tree.openElements.pop()
        else:
            self.startTagOther(token)

    def startTagForm(self, token):
        self.parser.parseError('unexpected-form-in-table')
        if self.tree.formPointer is None:
            self.tree.insertElement(token)
            self.tree.formPointer = self.tree.openElements[-1]
            self.tree.openElements.pop()

    def startTagOther(self, token):
        self.parser.parseError('unexpected-start-tag-implies-table-voodoo', {'name': token['name']})
        self.tree.insertFromTable = True
        self.parser.phases['inBody'].processStartTag(token)
        self.tree.insertFromTable = False

    def endTagTable(self, token):
        if self.tree.elementInScope('table', variant='table'):
            self.tree.generateImpliedEndTags()
            if self.tree.openElements[-1].name != 'table':
                self.parser.parseError('end-tag-too-early-named', {'gotName': 'table', 'expectedName': self.tree.openElements[-1].name})
            while self.tree.openElements[-1].name != 'table':
                self.tree.openElements.pop()
            self.tree.openElements.pop()
            self.parser.resetInsertionMode()
        else:
            assert self.parser.innerHTML
            self.parser.parseError()

    def endTagIgnore(self, token):
        self.parser.parseError('unexpected-end-tag', {'name': token['name']})

    def endTagOther(self, token):
        self.parser.parseError('unexpected-end-tag-implies-table-voodoo', {'name': token['name']})
        self.tree.insertFromTable = True
        self.parser.phases['inBody'].processEndTag(token)
        self.tree.insertFromTable = False
    startTagHandler = _utils.MethodDispatcher([('html', Phase.startTagHtml), ('caption', startTagCaption), ('colgroup', startTagColgroup), ('col', startTagCol), (('tbody', 'tfoot', 'thead'), startTagRowGroup), (('td', 'th', 'tr'), startTagImplyTbody), ('table', startTagTable), (('style', 'script'), startTagStyleScript), ('input', startTagInput), ('form', startTagForm)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([('table', endTagTable), (('body', 'caption', 'col', 'colgroup', 'html', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr'), endTagIgnore)])
    endTagHandler.default = endTagOther