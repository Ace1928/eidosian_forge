import sys
from io import StringIO
from typing import List, Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, succeed
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.trial.util import suppress as SUPPRESS
from twisted.web._element import UnexposedMethodError
from twisted.web.error import FlattenerError, MissingRenderMethod, MissingTemplateLoader
from twisted.web.iweb import IRequest, ITemplateLoader
from twisted.web.server import NOT_DONE_YET
from twisted.web.template import (
from twisted.web.test._util import FlattenTestCase
from twisted.web.test.test_web import DummyRequest
class FlattenIntegrationTests(FlattenTestCase):
    """
    Tests for integration between L{Element} and
    L{twisted.web._flatten.flatten}.
    """

    def test_roundTrip(self) -> None:
        """
        Given a series of parsable XML strings, verify that
        L{twisted.web._flatten.flatten} will flatten the L{Element} back to the
        input when sent on a round trip.
        """
        fragments = [b'<p>Hello, world.</p>', b'<p><!-- hello, world --></p>', b'<p><![CDATA[Hello, world.]]></p>', b'<test1 xmlns:test2="urn:test2"><test2:test3></test2:test3></test1>', b'<test1 xmlns="urn:test2"><test3></test3></test1>', b'<p>\xe2\x98\x83</p>']
        for xml in fragments:
            self.assertFlattensImmediately(Element(loader=XMLString(xml)), xml)

    def test_entityConversion(self) -> None:
        """
        When flattening an HTML entity, it should flatten out to the utf-8
        representation if possible.
        """
        element = Element(loader=XMLString('<p>&#9731;</p>'))
        self.assertFlattensImmediately(element, b'<p>\xe2\x98\x83</p>')

    def test_missingTemplateLoader(self) -> None:
        """
        Rendering an Element without a loader attribute raises the appropriate
        exception.
        """
        self.assertFlatteningRaises(Element(), MissingTemplateLoader)

    def test_missingRenderMethod(self) -> None:
        """
        Flattening an L{Element} with a C{loader} which has a tag with a render
        directive fails with L{FlattenerError} if there is no available render
        method to satisfy that directive.
        """
        element = Element(loader=XMLString('\n        <p xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1"\n          t:render="unknownMethod" />\n        '))
        self.assertFlatteningRaises(element, MissingRenderMethod)

    def test_transparentRendering(self) -> None:
        """
        A C{transparent} element should be eliminated from the DOM and rendered as
        only its children.
        """
        element = Element(loader=XMLString('<t:transparent xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1">Hello, world.</t:transparent>'))
        self.assertFlattensImmediately(element, b'Hello, world.')

    def test_attrRendering(self) -> None:
        """
        An Element with an attr tag renders the vaule of its attr tag as an
        attribute of its containing tag.
        """
        element = Element(loader=XMLString('<a xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1"><t:attr name="href">http://example.com</t:attr>Hello, world.</a>'))
        self.assertFlattensImmediately(element, b'<a href="http://example.com">Hello, world.</a>')

    def test_synchronousDeferredRecursion(self) -> None:
        """
        When rendering a large number of already-fired Deferreds we should not
        encounter any recursion errors or stack-depth issues.
        """
        self.assertFlattensImmediately([succeed('x') for i in range(250)], b'x' * 250)

    def test_errorToplevelAttr(self) -> None:
        """
        A template with a toplevel C{attr} tag will not load; it will raise
        L{AssertionError} if you try.
        """
        self.assertRaises(AssertionError, XMLString, "<t:attr\n            xmlns:t='http://twistedmatrix.com/ns/twisted.web.template/0.1'\n            name='something'\n            >hello</t:attr>\n            ")

    def test_errorUnnamedAttr(self) -> None:
        """
        A template with an C{attr} tag with no C{name} attribute will not load;
        it will raise L{AssertionError} if you try.
        """
        self.assertRaises(AssertionError, XMLString, "<html><t:attr\n            xmlns:t='http://twistedmatrix.com/ns/twisted.web.template/0.1'\n            >hello</t:attr></html>")

    def test_lenientPrefixBehavior(self) -> None:
        """
        If the parser sees a prefix it doesn't recognize on an attribute, it
        will pass it on through to serialization.
        """
        theInput = '<hello:world hello:sample="testing" xmlns:hello="http://made-up.example.com/ns/not-real">This is a made-up tag.</hello:world>'
        element = Element(loader=XMLString(theInput))
        self.assertFlattensTo(element, theInput.encode('utf8'))

    def test_deferredRendering(self) -> None:
        """
        An Element with a render method which returns a Deferred will render
        correctly.
        """

        class RenderfulElement(Element):

            @renderer
            def renderMethod(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
                return succeed('Hello, world.')
        element = RenderfulElement(loader=XMLString('\n        <p xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1"\n          t:render="renderMethod">\n            Goodbye, world.\n        </p>\n        '))
        self.assertFlattensImmediately(element, b'Hello, world.')

    def test_loaderClassAttribute(self) -> None:
        """
        If there is a non-None loader attribute on the class of an Element
        instance but none on the instance itself, the class attribute is used.
        """

        class SubElement(Element):
            loader = XMLString('<p>Hello, world.</p>')
        self.assertFlattensImmediately(SubElement(), b'<p>Hello, world.</p>')

    def test_directiveRendering(self) -> None:
        """
        An Element with a valid render directive has that directive invoked and
        the result added to the output.
        """
        renders = []

        class RenderfulElement(Element):

            @renderer
            def renderMethod(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
                renders.append((self, request))
                return tag('Hello, world.')
        element = RenderfulElement(loader=XMLString('\n        <p xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1"\n          t:render="renderMethod" />\n        '))
        self.assertFlattensImmediately(element, b'<p>Hello, world.</p>')

    def test_directiveRenderingOmittingTag(self) -> None:
        """
        An Element with a render method which omits the containing tag
        successfully removes that tag from the output.
        """

        class RenderfulElement(Element):

            @renderer
            def renderMethod(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
                return 'Hello, world.'
        element = RenderfulElement(loader=XMLString('\n        <p xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1"\n          t:render="renderMethod">\n            Goodbye, world.\n        </p>\n        '))
        self.assertFlattensImmediately(element, b'Hello, world.')

    def test_elementContainingStaticElement(self) -> None:
        """
        An Element which is returned by the render method of another Element is
        rendered properly.
        """

        class RenderfulElement(Element):

            @renderer
            def renderMethod(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
                return tag(Element(loader=XMLString('<em>Hello, world.</em>')))
        element = RenderfulElement(loader=XMLString('\n        <p xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1"\n          t:render="renderMethod" />\n        '))
        self.assertFlattensImmediately(element, b'<p><em>Hello, world.</em></p>')

    def test_elementUsingSlots(self) -> None:
        """
        An Element which is returned by the render method of another Element is
        rendered properly.
        """

        class RenderfulElement(Element):

            @renderer
            def renderMethod(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
                return tag.fillSlots(test2='world.')
        element = RenderfulElement(loader=XMLString('<p xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1" t:render="renderMethod"><t:slot name="test1" default="Hello, " /><t:slot name="test2" /></p>'))
        self.assertFlattensImmediately(element, b'<p>Hello, world.</p>')

    def test_elementContainingDynamicElement(self) -> None:
        """
        Directives in the document factory of an Element returned from a render
        method of another Element are satisfied from the correct object: the
        "inner" Element.
        """

        class OuterElement(Element):

            @renderer
            def outerMethod(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
                return tag(InnerElement(loader=XMLString('\n                <t:ignored\n                  xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1"\n                  t:render="innerMethod" />\n                ')))

        class InnerElement(Element):

            @renderer
            def innerMethod(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
                return 'Hello, world.'
        element = OuterElement(loader=XMLString('\n        <p xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1"\n          t:render="outerMethod" />\n        '))
        self.assertFlattensImmediately(element, b'<p>Hello, world.</p>')

    def test_sameLoaderTwice(self) -> None:
        """
        Rendering the output of a loader, or even the same element, should
        return different output each time.
        """
        sharedLoader = XMLString('<p xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1"><t:transparent t:render="classCounter" /> <t:transparent t:render="instanceCounter" /></p>')

        class DestructiveElement(Element):
            count = 0
            instanceCount = 0
            loader = sharedLoader

            @renderer
            def classCounter(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
                DestructiveElement.count += 1
                return tag(str(DestructiveElement.count))

            @renderer
            def instanceCounter(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
                self.instanceCount += 1
                return tag(str(self.instanceCount))
        e1 = DestructiveElement()
        e2 = DestructiveElement()
        self.assertFlattensImmediately(e1, b'<p>1 1</p>')
        self.assertFlattensImmediately(e1, b'<p>2 2</p>')
        self.assertFlattensImmediately(e2, b'<p>3 1</p>')