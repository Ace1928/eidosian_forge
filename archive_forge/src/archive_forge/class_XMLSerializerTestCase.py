import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
class XMLSerializerTestCase(unittest.TestCase):

    def test_with_xml_decl(self):
        stream = Stream([(Stream.XML_DECL, ('1.0', None, -1), (None, -1, -1))])
        output = stream.render(XMLSerializer, doctype='xhtml', encoding=None)
        self.assertEqual('<?xml version="1.0"?>\n<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n', output)

    def test_doctype_in_stream(self):
        stream = Stream([(Stream.DOCTYPE, DocType.HTML_STRICT, (None, -1, -1))])
        output = stream.render(XMLSerializer, encoding=None)
        self.assertEqual('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n', output)

    def test_doctype_in_stream_no_sysid(self):
        stream = Stream([(Stream.DOCTYPE, ('html', '-//W3C//DTD HTML 4.01//EN', None), (None, -1, -1))])
        output = stream.render(XMLSerializer, encoding=None)
        self.assertEqual('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN">\n', output)

    def test_doctype_in_stream_no_pubid(self):
        stream = Stream([(Stream.DOCTYPE, ('html', None, 'http://www.w3.org/TR/html4/strict.dtd'), (None, -1, -1))])
        output = stream.render(XMLSerializer, encoding=None)
        self.assertEqual('<!DOCTYPE html SYSTEM "http://www.w3.org/TR/html4/strict.dtd">\n', output)

    def test_doctype_in_stream_no_pubid_or_sysid(self):
        stream = Stream([(Stream.DOCTYPE, ('html', None, None), (None, -1, -1))])
        output = stream.render(XMLSerializer, encoding=None)
        self.assertEqual('<!DOCTYPE html>\n', output)

    def test_serializer_doctype(self):
        stream = Stream([])
        output = stream.render(XMLSerializer, doctype=DocType.HTML_STRICT, encoding=None)
        self.assertEqual('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n', output)

    def test_doctype_one_and_only(self):
        stream = Stream([(Stream.DOCTYPE, ('html', None, None), (None, -1, -1))])
        output = stream.render(XMLSerializer, doctype=DocType.HTML_STRICT, encoding=None)
        self.assertEqual('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n', output)

    def test_comment(self):
        stream = Stream([(Stream.COMMENT, 'foo bar', (None, -1, -1))])
        output = stream.render(XMLSerializer, encoding=None)
        self.assertEqual('<!--foo bar-->', output)

    def test_processing_instruction(self):
        stream = Stream([(Stream.PI, ('python', 'x = 2'), (None, -1, -1))])
        output = stream.render(XMLSerializer, encoding=None)
        self.assertEqual('<?python x = 2?>', output)

    def test_nested_default_namespaces(self):
        stream = Stream([(Stream.START_NS, ('', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}div'), Attrs()), (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}p'), Attrs()), (None, -1, -1)), (Stream.END, QName('http://example.org/}p'), (None, -1, -1)), (Stream.END_NS, '', (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}p'), Attrs()), (None, -1, -1)), (Stream.END, QName('http://example.org/}p'), (None, -1, -1)), (Stream.END_NS, '', (None, -1, -1)), (Stream.TEXT, '\n        ', (None, -1, -1)), (Stream.END, QName('http://example.org/}div'), (None, -1, -1)), (Stream.END_NS, '', (None, -1, -1))])
        output = stream.render(XMLSerializer, encoding=None)
        self.assertEqual('<div xmlns="http://example.org/">\n          <p/>\n          <p/>\n        </div>', output)

    def test_nested_bound_namespaces(self):
        stream = Stream([(Stream.START_NS, ('x', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}div'), Attrs()), (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('x', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}p'), Attrs()), (None, -1, -1)), (Stream.END, QName('http://example.org/}p'), (None, -1, -1)), (Stream.END_NS, 'x', (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('x', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}p'), Attrs()), (None, -1, -1)), (Stream.END, QName('http://example.org/}p'), (None, -1, -1)), (Stream.END_NS, 'x', (None, -1, -1)), (Stream.TEXT, '\n        ', (None, -1, -1)), (Stream.END, QName('http://example.org/}div'), (None, -1, -1)), (Stream.END_NS, 'x', (None, -1, -1))])
        output = stream.render(XMLSerializer, encoding=None)
        self.assertEqual('<x:div xmlns:x="http://example.org/">\n          <x:p/>\n          <x:p/>\n        </x:div>', output)

    def test_multiple_default_namespaces(self):
        stream = Stream([(Stream.START, (QName('div'), Attrs()), (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}p'), Attrs()), (None, -1, -1)), (Stream.END, QName('http://example.org/}p'), (None, -1, -1)), (Stream.END_NS, '', (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}p'), Attrs()), (None, -1, -1)), (Stream.END, QName('http://example.org/}p'), (None, -1, -1)), (Stream.END_NS, '', (None, -1, -1)), (Stream.TEXT, '\n        ', (None, -1, -1)), (Stream.END, QName('div'), (None, -1, -1))])
        output = stream.render(XMLSerializer, encoding=None)
        self.assertEqual('<div>\n          <p xmlns="http://example.org/"/>\n          <p xmlns="http://example.org/"/>\n        </div>', output)

    def test_multiple_bound_namespaces(self):
        stream = Stream([(Stream.START, (QName('div'), Attrs()), (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('x', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}p'), Attrs()), (None, -1, -1)), (Stream.END, QName('http://example.org/}p'), (None, -1, -1)), (Stream.END_NS, 'x', (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('x', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}p'), Attrs()), (None, -1, -1)), (Stream.END, QName('http://example.org/}p'), (None, -1, -1)), (Stream.END_NS, 'x', (None, -1, -1)), (Stream.TEXT, '\n        ', (None, -1, -1)), (Stream.END, QName('div'), (None, -1, -1))])
        output = stream.render(XMLSerializer, encoding=None)
        self.assertEqual('<div>\n          <x:p xmlns:x="http://example.org/"/>\n          <x:p xmlns:x="http://example.org/"/>\n        </div>', output)

    def test_atom_with_xhtml(self):
        text = '<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en">\n            <id>urn:uuid:c60843aa-0da8-4fa6-bbe5-98007bc6774e</id>\n            <updated>2007-01-28T11:36:02.807108-06:00</updated>\n            <title type="xhtml">\n                <div xmlns="http://www.w3.org/1999/xhtml">Example</div>\n            </title>\n            <subtitle type="xhtml">\n                <div xmlns="http://www.w3.org/1999/xhtml">Bla bla bla</div>\n            </subtitle>\n            <icon/>\n        </feed>'
        output = XML(text).render(XMLSerializer, encoding=None)
        self.assertEqual(text, output)