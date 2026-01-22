from itertools import chain
import codecs
from xml.parsers import expat
import six
from six.moves import html_entities as entities, html_parser as html
from genshi.core import Attrs, QName, Stream, stripentities
from genshi.core import START, END, XML_DECL, DOCTYPE, TEXT, START_NS, \
from genshi.compat import StringIO, BytesIO
class XMLParser(object):
    """Generator-based XML parser based on roughly equivalent code in
    Kid/ElementTree.
    
    The parsing is initiated by iterating over the parser object:
    
    >>> parser = XMLParser(StringIO('<root id="2"><child>Foo</child></root>'))
    >>> for kind, data, pos in parser:
    ...     print('%s %s' % (kind, data))
    START (QName('root'), Attrs([(QName('id'), '2')]))
    START (QName('child'), Attrs())
    TEXT Foo
    END child
    END root
    """
    _entitydefs = ['<!ENTITY %s "&#%d;">' % (name, value) for name, value in entities.name2codepoint.items()]
    _external_dtd = u'\n'.join(_entitydefs).encode('utf-8')

    def __init__(self, source, filename=None, encoding=None):
        """Initialize the parser for the given XML input.
        
        :param source: the XML text as a file-like object
        :param filename: the name of the file, if appropriate
        :param encoding: the encoding of the file; if not specified, the
                         encoding is assumed to be ASCII, UTF-8, or UTF-16, or
                         whatever the encoding specified in the XML declaration
                         (if any)
        """
        self.source = source
        self.filename = filename
        parser = expat.ParserCreate(encoding, '}')
        parser.buffer_text = True
        if hasattr(parser, 'returns_unicode'):
            parser.returns_unicode = True
        parser.ordered_attributes = True
        parser.StartElementHandler = self._handle_start
        parser.EndElementHandler = self._handle_end
        parser.CharacterDataHandler = self._handle_data
        parser.StartDoctypeDeclHandler = self._handle_doctype
        parser.StartNamespaceDeclHandler = self._handle_start_ns
        parser.EndNamespaceDeclHandler = self._handle_end_ns
        parser.StartCdataSectionHandler = self._handle_start_cdata
        parser.EndCdataSectionHandler = self._handle_end_cdata
        parser.ProcessingInstructionHandler = self._handle_pi
        parser.XmlDeclHandler = self._handle_xml_decl
        parser.CommentHandler = self._handle_comment
        parser.DefaultHandler = self._handle_other
        parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_ALWAYS)
        parser.UseForeignDTD()
        parser.ExternalEntityRefHandler = self._build_foreign
        self.expat = parser
        self._queue = []

    def parse(self):
        """Generator that parses the XML source, yielding markup events.
        
        :return: a markup event stream
        :raises ParseError: if the XML text is not well formed
        """

        def _generate():
            try:
                bufsize = 4 * 1024
                done = False
                while 1:
                    while not done and len(self._queue) == 0:
                        data = self.source.read(bufsize)
                        if not data:
                            if hasattr(self, 'expat'):
                                self.expat.Parse('', True)
                                del self.expat
                            done = True
                        else:
                            if isinstance(data, six.text_type):
                                data = data.encode('utf-8')
                            self.expat.Parse(data, False)
                    for event in self._queue:
                        yield event
                    self._queue = []
                    if done:
                        break
            except expat.ExpatError as e:
                msg = str(e)
                raise ParseError(msg, self.filename, e.lineno, e.offset)
        return Stream(_generate()).filter(_coalesce)

    def __iter__(self):
        return iter(self.parse())

    def _build_foreign(self, context, base, sysid, pubid):
        parser = self.expat.ExternalEntityParserCreate(context)
        parser.ParseFile(BytesIO(self._external_dtd))
        return 1

    def _enqueue(self, kind, data=None, pos=None):
        if pos is None:
            pos = self._getpos()
        if kind is TEXT:
            if '\n' in data:
                lines = data.splitlines()
                lineno = pos[1] - len(lines) + 1
                offset = -1
            else:
                lineno = pos[1]
                offset = pos[2] - len(data)
            pos = (pos[0], lineno, offset)
        self._queue.append((kind, data, pos))

    def _getpos_unknown(self):
        return (self.filename, -1, -1)

    def _getpos(self):
        return (self.filename, self.expat.CurrentLineNumber, self.expat.CurrentColumnNumber)

    def _handle_start(self, tag, attrib):
        attrs = Attrs([(QName(name), value) for name, value in zip(*[iter(attrib)] * 2)])
        self._enqueue(START, (QName(tag), attrs))

    def _handle_end(self, tag):
        self._enqueue(END, QName(tag))

    def _handle_data(self, text):
        self._enqueue(TEXT, text)

    def _handle_xml_decl(self, version, encoding, standalone):
        self._enqueue(XML_DECL, (version, encoding, standalone))

    def _handle_doctype(self, name, sysid, pubid, has_internal_subset):
        self._enqueue(DOCTYPE, (name, pubid, sysid))

    def _handle_start_ns(self, prefix, uri):
        self._enqueue(START_NS, (prefix or '', uri))

    def _handle_end_ns(self, prefix):
        self._enqueue(END_NS, prefix or '')

    def _handle_start_cdata(self):
        self._enqueue(START_CDATA)

    def _handle_end_cdata(self):
        self._enqueue(END_CDATA)

    def _handle_pi(self, target, data):
        self._enqueue(PI, (target, data))

    def _handle_comment(self, text):
        self._enqueue(COMMENT, text)

    def _handle_other(self, text):
        if text.startswith('&'):
            try:
                text = six.unichr(entities.name2codepoint[text[1:-1]])
                self._enqueue(TEXT, text)
            except KeyError:
                filename, lineno, offset = self._getpos()
                error = expat.error('undefined entity "%s": line %d, column %d' % (text, lineno, offset))
                error.code = expat.errors.XML_ERROR_UNDEFINED_ENTITY
                error.lineno = lineno
                error.offset = offset
                raise error