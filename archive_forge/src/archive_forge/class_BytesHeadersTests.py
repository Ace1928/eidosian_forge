from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
class BytesHeadersTests(TestCase):
    """
    Tests for L{Headers}, using L{bytes} arguments for methods.
    """

    def test_sanitizeLinearWhitespace(self) -> None:
        """
        Linear whitespace in header names or values is replaced with a
        single space.
        """
        assertSanitized(self, bytesLinearWhitespaceComponents, sanitizedBytes)

    def test_initializer(self) -> None:
        """
        The header values passed to L{Headers.__init__} can be retrieved via
        L{Headers.getRawHeaders}.
        """
        h = Headers({b'Foo': [b'bar']})
        self.assertEqual(h.getRawHeaders(b'foo'), [b'bar'])

    def test_setRawHeaders(self) -> None:
        """
        L{Headers.setRawHeaders} sets the header values for the given
        header name to the sequence of byte string values.
        """
        rawValue = [b'value1', b'value2']
        h = Headers()
        h.setRawHeaders(b'test', rawValue)
        self.assertTrue(h.hasHeader(b'test'))
        self.assertTrue(h.hasHeader(b'Test'))
        self.assertEqual(h.getRawHeaders(b'test'), rawValue)

    def test_rawHeadersTypeCheckingValuesIterable(self) -> None:
        """
        L{Headers.setRawHeaders} requires values to be of type list.
        """
        h = Headers()
        self.assertRaises(TypeError, h.setRawHeaders, b'key', {b'Foo': b'bar'})

    def test_rawHeadersTypeCheckingName(self) -> None:
        """
        L{Headers.setRawHeaders} requires C{name} to be a L{bytes} or
        L{str} string.
        """
        h = Headers()
        e = self.assertRaises(TypeError, h.setRawHeaders, None, [b'foo'])
        self.assertEqual(e.args[0], "Header name is an instance of <class 'NoneType'>, not bytes or str")

    def test_rawHeadersTypeCheckingValuesAreString(self) -> None:
        """
        L{Headers.setRawHeaders} requires values to a L{list} of L{bytes} or
        L{str} strings.
        """
        h = Headers()
        e = self.assertRaises(TypeError, h.setRawHeaders, b'key', [b'bar', None])
        self.assertEqual(e.args[0], "Header value at position 1 is an instance of <class 'NoneType'>, not bytes or str")

    def test_addRawHeader(self) -> None:
        """
        L{Headers.addRawHeader} adds a new value for a given header.
        """
        h = Headers()
        h.addRawHeader(b'test', b'lemur')
        self.assertEqual(h.getRawHeaders(b'test'), [b'lemur'])
        h.addRawHeader(b'test', b'panda')
        self.assertEqual(h.getRawHeaders(b'test'), [b'lemur', b'panda'])

    def test_addRawHeaderTypeCheckName(self) -> None:
        """
        L{Headers.addRawHeader} requires C{name} to be a L{bytes} or L{str}
        string.
        """
        h = Headers()
        e = self.assertRaises(TypeError, h.addRawHeader, None, b'foo')
        self.assertEqual(e.args[0], "Header name is an instance of <class 'NoneType'>, not bytes or str")

    def test_addRawHeaderTypeCheckValue(self) -> None:
        """
        L{Headers.addRawHeader} requires value to be a L{bytes} or L{str}
        string.
        """
        h = Headers()
        e = self.assertRaises(TypeError, h.addRawHeader, b'key', None)
        self.assertEqual(e.args[0], "Header value is an instance of <class 'NoneType'>, not bytes or str")

    def test_getRawHeadersNoDefault(self) -> None:
        """
        L{Headers.getRawHeaders} returns L{None} if the header is not found and
        no default is specified.
        """
        self.assertIsNone(Headers().getRawHeaders(b'test'))

    def test_getRawHeadersDefaultValue(self) -> None:
        """
        L{Headers.getRawHeaders} returns the specified default value when no
        header is found.
        """
        h = Headers()
        default = object()
        self.assertIdentical(h.getRawHeaders(b'test', default), default)

    def test_getRawHeadersWithDefaultMatchingValue(self) -> None:
        """
        If the object passed as the value list to L{Headers.setRawHeaders}
        is later passed as a default to L{Headers.getRawHeaders}, the
        result nevertheless contains encoded values.
        """
        h = Headers()
        default = ['value']
        h.setRawHeaders(b'key', default)
        self.assertIsInstance(h.getRawHeaders(b'key', default)[0], bytes)
        self.assertEqual(h.getRawHeaders(b'key', default), [b'value'])

    def test_getRawHeaders(self) -> None:
        """
        L{Headers.getRawHeaders} returns the values which have been set for a
        given header.
        """
        h = Headers()
        h.setRawHeaders(b'test', [b'lemur'])
        self.assertEqual(h.getRawHeaders(b'test'), [b'lemur'])
        self.assertEqual(h.getRawHeaders(b'Test'), [b'lemur'])

    def test_hasHeaderTrue(self) -> None:
        """
        Check that L{Headers.hasHeader} returns C{True} when the given header
        is found.
        """
        h = Headers()
        h.setRawHeaders(b'test', [b'lemur'])
        self.assertTrue(h.hasHeader(b'test'))
        self.assertTrue(h.hasHeader(b'Test'))

    def test_hasHeaderFalse(self) -> None:
        """
        L{Headers.hasHeader} returns C{False} when the given header is not
        found.
        """
        self.assertFalse(Headers().hasHeader(b'test'))

    def test_removeHeader(self) -> None:
        """
        Check that L{Headers.removeHeader} removes the given header.
        """
        h = Headers()
        h.setRawHeaders(b'foo', [b'lemur'])
        self.assertTrue(h.hasHeader(b'foo'))
        h.removeHeader(b'foo')
        self.assertFalse(h.hasHeader(b'foo'))
        h.setRawHeaders(b'bar', [b'panda'])
        self.assertTrue(h.hasHeader(b'bar'))
        h.removeHeader(b'Bar')
        self.assertFalse(h.hasHeader(b'bar'))

    def test_removeHeaderDoesntExist(self) -> None:
        """
        L{Headers.removeHeader} is a no-operation when the specified header is
        not found.
        """
        h = Headers()
        h.removeHeader(b'test')
        self.assertEqual(list(h.getAllRawHeaders()), [])

    def test_canonicalNameCaps(self) -> None:
        """
        L{Headers._canonicalNameCaps} returns the canonical capitalization for
        the given header.
        """
        h = Headers()
        self.assertEqual(h._canonicalNameCaps(b'test'), b'Test')
        self.assertEqual(h._canonicalNameCaps(b'test-stuff'), b'Test-Stuff')
        self.assertEqual(h._canonicalNameCaps(b'content-md5'), b'Content-MD5')
        self.assertEqual(h._canonicalNameCaps(b'dnt'), b'DNT')
        self.assertEqual(h._canonicalNameCaps(b'etag'), b'ETag')
        self.assertEqual(h._canonicalNameCaps(b'p3p'), b'P3P')
        self.assertEqual(h._canonicalNameCaps(b'te'), b'TE')
        self.assertEqual(h._canonicalNameCaps(b'www-authenticate'), b'WWW-Authenticate')
        self.assertEqual(h._canonicalNameCaps(b'x-xss-protection'), b'X-XSS-Protection')

    def test_getAllRawHeaders(self) -> None:
        """
        L{Headers.getAllRawHeaders} returns an iterable of (k, v) pairs, where
        C{k} is the canonicalized representation of the header name, and C{v}
        is a sequence of values.
        """
        h = Headers()
        h.setRawHeaders(b'test', [b'lemurs'])
        h.setRawHeaders(b'www-authenticate', [b'basic aksljdlk='])
        allHeaders = {(k, tuple(v)) for k, v in h.getAllRawHeaders()}
        self.assertEqual(allHeaders, {(b'WWW-Authenticate', (b'basic aksljdlk=',)), (b'Test', (b'lemurs',))})

    def test_headersComparison(self) -> None:
        """
        A L{Headers} instance compares equal to itself and to another
        L{Headers} instance with the same values.
        """
        first = Headers()
        first.setRawHeaders(b'foo', [b'panda'])
        second = Headers()
        second.setRawHeaders(b'foo', [b'panda'])
        third = Headers()
        third.setRawHeaders(b'foo', [b'lemur', b'panda'])
        self.assertEqual(first, first)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_otherComparison(self) -> None:
        """
        An instance of L{Headers} does not compare equal to other unrelated
        objects.
        """
        h = Headers()
        self.assertNotEqual(h, ())
        self.assertNotEqual(h, object())
        self.assertNotEqual(h, b'foo')

    def test_repr(self) -> None:
        """
        The L{repr} of a L{Headers} instance shows the names and values of all
        the headers it contains.
        """
        foo = b'foo'
        bar = b'bar'
        baz = b'baz'
        self.assertEqual(repr(Headers({foo: [bar, baz]})), f'Headers({{{foo!r}: [{bar!r}, {baz!r}]}})')

    def test_reprWithRawBytes(self) -> None:
        """
        The L{repr} of a L{Headers} instance shows the names and values of all
        the headers it contains, not attempting to decode any raw bytes.
        """
        foo = b'foo'
        bar = b'bar\xe1'
        baz = b'baz\xe1'
        self.assertEqual(repr(Headers({foo: [bar, baz]})), f'Headers({{{foo!r}: [{bar!r}, {baz!r}]}})')

    def test_subclassRepr(self) -> None:
        """
        The L{repr} of an instance of a subclass of L{Headers} uses the name
        of the subclass instead of the string C{"Headers"}.
        """
        foo = b'foo'
        bar = b'bar'
        baz = b'baz'

        class FunnyHeaders(Headers):
            pass
        self.assertEqual(repr(FunnyHeaders({foo: [bar, baz]})), f'FunnyHeaders({{{foo!r}: [{bar!r}, {baz!r}]}})')

    def test_copy(self) -> None:
        """
        L{Headers.copy} creates a new independent copy of an existing
        L{Headers} instance, allowing future modifications without impacts
        between the copies.
        """
        h = Headers()
        h.setRawHeaders(b'test', [b'foo'])
        i = h.copy()
        self.assertEqual(i.getRawHeaders(b'test'), [b'foo'])
        h.addRawHeader(b'test', b'bar')
        self.assertEqual(i.getRawHeaders(b'test'), [b'foo'])
        i.addRawHeader(b'test', b'baz')
        self.assertEqual(h.getRawHeaders(b'test'), [b'foo', b'bar'])