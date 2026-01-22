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
class _StackElement(Element):
    """
    L{_StackElement} renders an L{IRenderable} which can render a list of frames.
    """

    def __init__(self, loader, stackFrames):
        Element.__init__(self, loader)
        self.stackFrames = stackFrames

    @renderer
    def frames(self, request, tag):
        """
        Render the list of frames in this L{_StackElement}, replacing C{tag}.
        """
        return [_FrameElement(TagLoader(tag.clone()), frame) for frame in self.stackFrames]