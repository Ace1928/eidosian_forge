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
class FailureElement(Element):
    """
    L{FailureElement} is an L{IRenderable} which can render detailed information
    about a L{Failure<twisted.python.failure.Failure>}.

    @ivar failure: The L{Failure<twisted.python.failure.Failure>} instance which
        will be rendered.

    @since: 12.1
    """
    loader = XMLString('\n<div xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1">\n  <style type="text/css">\n    div.error {\n      color: red;\n      font-family: Verdana, Arial, helvetica, sans-serif;\n      font-weight: bold;\n    }\n\n    div {\n      font-family: Verdana, Arial, helvetica, sans-serif;\n    }\n\n    div.stackTrace {\n    }\n\n    div.frame {\n      padding: 1em;\n      background: white;\n      border-bottom: thin black dashed;\n    }\n\n    div.frame:first-child {\n      padding: 1em;\n      background: white;\n      border-top: thin black dashed;\n      border-bottom: thin black dashed;\n    }\n\n    div.location {\n    }\n\n    span.function {\n      font-weight: bold;\n      font-family: "Courier New", courier, monospace;\n    }\n\n    div.snippet {\n      margin-bottom: 0.5em;\n      margin-left: 1em;\n      background: #FFFFDD;\n    }\n\n    div.snippetHighlightLine {\n      color: red;\n    }\n\n    span.code {\n      font-family: "Courier New", courier, monospace;\n    }\n  </style>\n\n  <div class="error">\n    <span t:render="type" />: <span t:render="value" />\n  </div>\n  <div class="stackTrace" t:render="traceback">\n    <div class="frame" t:render="frames">\n      <div class="location">\n        <span t:render="filename" />:<span t:render="lineNumber" /> in\n        <span class="function" t:render="function" />\n      </div>\n      <div class="snippet" t:render="source">\n        <div t:render="sourceLines">\n          <span class="lineno" t:render="lineNumber" />\n          <code class="code" t:render="sourceLine" />\n        </div>\n      </div>\n    </div>\n  </div>\n  <div class="error">\n    <span t:render="type" />: <span t:render="value" />\n  </div>\n</div>\n')

    def __init__(self, failure, loader=None):
        Element.__init__(self, loader)
        self.failure = failure

    @renderer
    def type(self, request, tag):
        """
        Render the exception type as a child of C{tag}.
        """
        return tag(fullyQualifiedName(self.failure.type))

    @renderer
    def value(self, request, tag):
        """
        Render the exception value as a child of C{tag}.
        """
        return tag(str(self.failure.value).encode('utf8'))

    @renderer
    def traceback(self, request, tag):
        """
        Render all the frames in the wrapped
        L{Failure<twisted.python.failure.Failure>}'s traceback stack, replacing
        C{tag}.
        """
        return _StackElement(TagLoader(tag), self.failure.frames)