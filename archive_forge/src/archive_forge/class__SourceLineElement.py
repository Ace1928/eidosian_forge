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
class _SourceLineElement(Element):
    """
    L{_SourceLineElement} is an L{IRenderable} which can render a single line of
    source code.

    @ivar number: A C{int} giving the line number of the source code to be
        rendered.
    @ivar source: A C{str} giving the source code to be rendered.
    """

    def __init__(self, loader, number, source):
        Element.__init__(self, loader)
        self.number = number
        self.source = source

    @renderer
    def sourceLine(self, request, tag):
        """
        Render the line of source as a child of C{tag}.
        """
        return tag(self.source.replace('  ', ' \xa0'))

    @renderer
    def lineNumber(self, request, tag):
        """
        Render the line number as a child of C{tag}.
        """
        return tag(str(self.number))