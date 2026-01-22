from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
class MicroDOMParser(XMLParser):
    soonClosers = 'area link br img hr input base meta'.split()
    laterClosers = {'p': ['p', 'dt'], 'dt': ['dt', 'dd'], 'dd': ['dt', 'dd'], 'li': ['li'], 'tbody': ['thead', 'tfoot', 'tbody'], 'thead': ['thead', 'tfoot', 'tbody'], 'tfoot': ['thead', 'tfoot', 'tbody'], 'colgroup': ['colgroup'], 'col': ['col'], 'tr': ['tr'], 'td': ['td'], 'th': ['th'], 'head': ['body'], 'title': ['head', 'body'], 'option': ['option']}

    def __init__(self, beExtremelyLenient=0, caseInsensitive=1, preserveCase=0, soonClosers=soonClosers, laterClosers=laterClosers):
        self.elementstack = []
        d = {'xmlns': 'xmlns', '': None}
        dr = _reverseDict(d)
        self.nsstack = [(d, None, dr)]
        self.documents = []
        self._mddoctype = None
        self.beExtremelyLenient = beExtremelyLenient
        self.caseInsensitive = caseInsensitive
        self.preserveCase = preserveCase or not caseInsensitive
        self.soonClosers = soonClosers
        self.laterClosers = laterClosers

    def shouldPreserveSpace(self):
        for edx in range(len(self.elementstack)):
            el = self.elementstack[-edx]
            if el.tagName == 'pre' or el.getAttribute('xml:space', '') == 'preserve':
                return 1
        return 0

    def _getparent(self):
        if self.elementstack:
            return self.elementstack[-1]
        else:
            return None
    COMMENT = re.compile('\\s*/[/*]\\s*')

    def _fixScriptElement(self, el):
        if not self.beExtremelyLenient or not len(el.childNodes) == 1:
            return
        c = el.firstChild()
        if isinstance(c, Text):
            prefix = ''
            oldvalue = c.value
            match = self.COMMENT.match(oldvalue)
            if match:
                prefix = match.group()
                oldvalue = oldvalue[len(prefix):]
            try:
                e = parseString('<a>%s</a>' % oldvalue).childNodes[0]
            except (ParseError, MismatchedTags):
                return
            if len(e.childNodes) != 1:
                return
            e = e.firstChild()
            if isinstance(e, (CDATASection, Comment)):
                el.childNodes = []
                if prefix:
                    el.childNodes.append(Text(prefix))
                el.childNodes.append(e)

    def gotDoctype(self, doctype):
        self._mddoctype = doctype

    def gotTagStart(self, name, attributes):
        parent = self._getparent()
        if self.beExtremelyLenient and isinstance(parent, Element):
            parentName = parent.tagName
            myName = name
            if self.caseInsensitive:
                parentName = parentName.lower()
                myName = myName.lower()
            if myName in self.laterClosers.get(parentName, []):
                self.gotTagEnd(parent.tagName)
                parent = self._getparent()
        attributes = _unescapeDict(attributes)
        namespaces = self.nsstack[-1][0]
        newspaces = {}
        keysToDelete = []
        for k, v in attributes.items():
            if k.startswith('xmlns'):
                spacenames = k.split(':', 1)
                if len(spacenames) == 2:
                    newspaces[spacenames[1]] = v
                else:
                    newspaces[''] = v
                keysToDelete.append(k)
        for k in keysToDelete:
            del attributes[k]
        if newspaces:
            namespaces = namespaces.copy()
            namespaces.update(newspaces)
        keysToDelete = []
        for k, v in attributes.items():
            ksplit = k.split(':', 1)
            if len(ksplit) == 2:
                pfx, tv = ksplit
                if pfx != 'xml' and pfx in namespaces:
                    attributes[namespaces[pfx], tv] = v
                    keysToDelete.append(k)
        for k in keysToDelete:
            del attributes[k]
        el = Element(name, attributes, parent, self.filename, self.saveMark(), caseInsensitive=self.caseInsensitive, preserveCase=self.preserveCase, namespace=namespaces.get(''))
        revspaces = _reverseDict(newspaces)
        el.addPrefixes(revspaces)
        if newspaces:
            rscopy = self.nsstack[-1][2].copy()
            rscopy.update(revspaces)
            self.nsstack.append((namespaces, el, rscopy))
        self.elementstack.append(el)
        if parent:
            parent.appendChild(el)
        if self.beExtremelyLenient and el.tagName in self.soonClosers:
            self.gotTagEnd(name)

    def _gotStandalone(self, factory, data):
        parent = self._getparent()
        te = factory(data, parent)
        if parent:
            parent.appendChild(te)
        elif self.beExtremelyLenient:
            self.documents.append(te)

    def gotText(self, data):
        if data.strip() or self.shouldPreserveSpace():
            self._gotStandalone(Text, data)

    def gotComment(self, data):
        self._gotStandalone(Comment, data)

    def gotEntityReference(self, entityRef):
        self._gotStandalone(EntityReference, entityRef)

    def gotCData(self, cdata):
        self._gotStandalone(CDATASection, cdata)

    def gotTagEnd(self, name):
        if not self.elementstack:
            if self.beExtremelyLenient:
                return
            raise MismatchedTags(*(self.filename, 'NOTHING', name) + self.saveMark() + (0, 0))
        el = self.elementstack.pop()
        pfxdix = self.nsstack[-1][2]
        if self.nsstack[-1][1] is el:
            nstuple = self.nsstack.pop()
        else:
            nstuple = None
        if self.caseInsensitive:
            tn = el.tagName.lower()
            cname = name.lower()
        else:
            tn = el.tagName
            cname = name
        nsplit = name.split(':', 1)
        if len(nsplit) == 2:
            pfx, newname = nsplit
            ns = pfxdix.get(pfx, None)
            if ns is not None:
                if el.namespace != ns:
                    if not self.beExtremelyLenient:
                        raise MismatchedTags(*(self.filename, el.tagName, name) + self.saveMark() + el._markpos)
        if not tn == cname:
            if self.beExtremelyLenient:
                if self.elementstack:
                    lastEl = self.elementstack[0]
                    for idx in range(len(self.elementstack)):
                        if self.elementstack[-(idx + 1)].tagName == cname:
                            self.elementstack[-(idx + 1)].endTag(name)
                            break
                    else:
                        self.elementstack.append(el)
                        if nstuple is not None:
                            self.nsstack.append(nstuple)
                        return
                    del self.elementstack[-(idx + 1):]
                    if not self.elementstack:
                        self.documents.append(lastEl)
                        return
            else:
                raise MismatchedTags(*(self.filename, el.tagName, name) + self.saveMark() + el._markpos)
        el.endTag(name)
        if not self.elementstack:
            self.documents.append(el)
        if self.beExtremelyLenient and el.tagName == 'script':
            self._fixScriptElement(el)

    def connectionLost(self, reason):
        XMLParser.connectionLost(self, reason)
        if self.elementstack:
            if self.beExtremelyLenient:
                self.documents.append(self.elementstack[0])
            else:
                raise MismatchedTags(*(self.filename, self.elementstack[-1], 'END_OF_FILE') + self.saveMark() + self.elementstack[-1]._markpos)