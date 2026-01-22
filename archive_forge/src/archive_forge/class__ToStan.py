import io
import linecache
import warnings
from collections import OrderedDict
from html import escape
from typing import (
from xml.sax import handler, make_parser
from xml.sax.xmlreader import AttributesNSImpl, Locator
from zope.interface import implementer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import urlpath
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.reflect import fullyQualifiedName
from twisted.web import resource
from twisted.web._element import Element, renderer
from twisted.web._flatten import Flattenable, flatten, flattenString
from twisted.web._stan import CDATA, Comment, Tag, slot
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
class _ToStan(handler.ContentHandler, handler.EntityResolver):
    """
    A SAX parser which converts an XML document to the Twisted STAN
    Document Object Model.
    """

    def __init__(self, sourceFilename: Optional[str]):
        """
        @param sourceFilename: the filename the XML was loaded out of.
        """
        self.sourceFilename = sourceFilename
        self.prefixMap = _NSContext()
        self.inCDATA = False

    def setDocumentLocator(self, locator: Locator) -> None:
        """
        Set the document locator, which knows about line and character numbers.
        """
        self.locator = locator

    def startDocument(self) -> None:
        """
        Initialise the document.
        """
        self.document: List[Any] = []
        self.current = self.document
        self.stack: List[Any] = []
        self.xmlnsAttrs: List[Tuple[str, str]] = []

    def endDocument(self) -> None:
        """
        Document ended.
        """

    def processingInstruction(self, target: str, data: str) -> None:
        """
        Processing instructions are ignored.
        """

    def startPrefixMapping(self, prefix: Optional[str], uri: str) -> None:
        """
        Set up the prefix mapping, which maps fully qualified namespace URIs
        onto namespace prefixes.

        This gets called before startElementNS whenever an C{xmlns} attribute
        is seen.
        """
        self.prefixMap = _NSContext(self.prefixMap)
        self.prefixMap[uri] = prefix
        if uri == TEMPLATE_NAMESPACE:
            return
        if prefix is None:
            self.xmlnsAttrs.append(('xmlns', uri))
        else:
            self.xmlnsAttrs.append(('xmlns:%s' % prefix, uri))

    def endPrefixMapping(self, prefix: Optional[str]) -> None:
        """
        "Pops the stack" on the prefix mapping.

        Gets called after endElementNS.
        """
        parent = self.prefixMap.parent
        assert parent is not None, 'More prefix mapping ends than starts'
        self.prefixMap = parent

    def startElementNS(self, namespaceAndName: Tuple[str, str], qname: Optional[str], attrs: AttributesNSImpl) -> None:
        """
        Gets called when we encounter a new xmlns attribute.

        @param namespaceAndName: a (namespace, name) tuple, where name
            determines which type of action to take, if the namespace matches
            L{TEMPLATE_NAMESPACE}.
        @param qname: ignored.
        @param attrs: attributes on the element being started.
        """
        filename = self.sourceFilename
        lineNumber = self.locator.getLineNumber()
        columnNumber = self.locator.getColumnNumber()
        ns, name = namespaceAndName
        if ns == TEMPLATE_NAMESPACE:
            if name == 'transparent':
                name = ''
            elif name == 'slot':
                default: Optional[str]
                try:
                    default = attrs[None, 'default']
                except KeyError:
                    default = None
                sl = slot(attrs[None, 'name'], default=default, filename=filename, lineNumber=lineNumber, columnNumber=columnNumber)
                self.stack.append(sl)
                self.current.append(sl)
                self.current = sl.children
                return
        render = None
        ordered = OrderedDict(attrs)
        for k, v in list(ordered.items()):
            attrNS, justTheName = k
            if attrNS != TEMPLATE_NAMESPACE:
                continue
            if justTheName == 'render':
                render = v
                del ordered[k]
        nonTemplateAttrs = OrderedDict()
        for (attrNs, attrName), v in ordered.items():
            nsPrefix = self.prefixMap.get(attrNs)
            if nsPrefix is None:
                attrKey = attrName
            else:
                attrKey = f'{nsPrefix}:{attrName}'
            nonTemplateAttrs[attrKey] = v
        if ns == TEMPLATE_NAMESPACE and name == 'attr':
            if not self.stack:
                raise AssertionError(f'<{{{TEMPLATE_NAMESPACE}}}attr> as top-level element')
            if 'name' not in nonTemplateAttrs:
                raise AssertionError(f'<{{{TEMPLATE_NAMESPACE}}}attr> requires a name attribute')
            el = Tag('', render=render, filename=filename, lineNumber=lineNumber, columnNumber=columnNumber)
            self.stack[-1].attributes[nonTemplateAttrs['name']] = el
            self.stack.append(el)
            self.current = el.children
            return
        if self.xmlnsAttrs:
            nonTemplateAttrs.update(OrderedDict(self.xmlnsAttrs))
            self.xmlnsAttrs = []
        if ns != TEMPLATE_NAMESPACE and ns is not None:
            prefix = self.prefixMap[ns]
            if prefix is not None:
                name = f'{self.prefixMap[ns]}:{name}'
        el = Tag(name, attributes=OrderedDict(cast(Mapping[Union[bytes, str], str], nonTemplateAttrs)), render=render, filename=filename, lineNumber=lineNumber, columnNumber=columnNumber)
        self.stack.append(el)
        self.current.append(el)
        self.current = el.children

    def characters(self, ch: str) -> None:
        """
        Called when we receive some characters.  CDATA characters get passed
        through as is.
        """
        if self.inCDATA:
            self.stack[-1].append(ch)
            return
        self.current.append(ch)

    def endElementNS(self, name: Tuple[str, str], qname: Optional[str]) -> None:
        """
        A namespace tag is closed.  Pop the stack, if there's anything left in
        it, otherwise return to the document's namespace.
        """
        self.stack.pop()
        if self.stack:
            self.current = self.stack[-1].children
        else:
            self.current = self.document

    def startDTD(self, name: str, publicId: str, systemId: str) -> None:
        """
        DTDs are ignored.
        """

    def endDTD(self, *args: object) -> None:
        """
        DTDs are ignored.
        """

    def startCDATA(self) -> None:
        """
        We're starting to be in a CDATA element, make a note of this.
        """
        self.inCDATA = True
        self.stack.append([])

    def endCDATA(self) -> None:
        """
        We're no longer in a CDATA element.  Collect up the characters we've
        parsed and put them in a new CDATA object.
        """
        self.inCDATA = False
        comment = ''.join(self.stack.pop())
        self.current.append(CDATA(comment))

    def comment(self, content: str) -> None:
        """
        Add an XML comment which we've encountered.
        """
        self.current.append(Comment(content))