from itertools import chain
import re
import six
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
class XHTMLSerializer(XMLSerializer):
    """Produces XHTML text from an event stream.
    
    >>> from genshi.builder import tag
    >>> elem = tag.div(tag.a(href='foo'), tag.br, tag.hr(noshade=True))
    >>> print(''.join(XHTMLSerializer()(elem.generate())))
    <div><a href="foo"></a><br /><hr noshade="noshade" /></div>
    """
    _EMPTY_ELEMS = frozenset(['area', 'base', 'basefont', 'br', 'col', 'frame', 'hr', 'img', 'input', 'isindex', 'link', 'meta', 'param'])
    _BOOLEAN_ATTRS = frozenset(['selected', 'checked', 'compact', 'declare', 'defer', 'disabled', 'ismap', 'multiple', 'nohref', 'noresize', 'noshade', 'nowrap', 'autofocus', 'readonly', 'required', 'formnovalidate'])
    _PRESERVE_SPACE = frozenset([QName('pre'), QName('http://www.w3.org/1999/xhtml}pre'), QName('textarea'), QName('http://www.w3.org/1999/xhtml}textarea')])

    def __init__(self, doctype=None, strip_whitespace=True, namespace_prefixes=None, drop_xml_decl=True, cache=True):
        super(XHTMLSerializer, self).__init__(doctype, False)
        self.filters = [EmptyTagFilter()]
        if strip_whitespace:
            self.filters.append(WhitespaceFilter(self._PRESERVE_SPACE))
        namespace_prefixes = namespace_prefixes or {}
        namespace_prefixes['http://www.w3.org/1999/xhtml'] = ''
        self.filters.append(NamespaceFlattener(prefixes=namespace_prefixes, cache=cache))
        if doctype:
            self.filters.append(DocTypeInserter(doctype))
        self.drop_xml_decl = drop_xml_decl
        self.cache = cache

    def __call__(self, stream):
        boolean_attrs = self._BOOLEAN_ATTRS
        empty_elems = self._EMPTY_ELEMS
        drop_xml_decl = self.drop_xml_decl
        have_decl = have_doctype = False
        in_cdata = False
        _emit, _get = self._prepare_cache()
        for filter_ in self.filters:
            stream = filter_(stream)
        for kind, data, pos in stream:
            if kind is TEXT and isinstance(data, Markup):
                yield data
                continue
            cached = _get((kind, data))
            if cached is not None:
                yield cached
            elif kind is START or kind is EMPTY:
                tag, attrib = data
                buf = ['<', tag]
                for attr, value in attrib:
                    if attr in boolean_attrs:
                        value = attr
                    elif attr == 'xml:lang' and 'lang' not in attrib:
                        buf += [' lang="', escape(value), '"']
                    elif attr == 'xml:space':
                        continue
                    buf += [' ', attr, '="', escape(value), '"']
                if kind is EMPTY:
                    if tag in empty_elems:
                        buf.append(' />')
                    else:
                        buf.append('></%s>' % tag)
                else:
                    buf.append('>')
                yield _emit(kind, data, Markup(''.join(buf)))
            elif kind is END:
                yield _emit(kind, data, Markup('</%s>' % data))
            elif kind is TEXT:
                if in_cdata:
                    yield _emit(kind, data, data)
                else:
                    yield _emit(kind, data, escape(data, quotes=False))
            elif kind is COMMENT:
                yield _emit(kind, data, Markup('<!--%s-->' % data))
            elif kind is DOCTYPE and (not have_doctype):
                name, pubid, sysid = data
                buf = ['<!DOCTYPE %s']
                if pubid:
                    buf.append(' PUBLIC "%s"')
                elif sysid:
                    buf.append(' SYSTEM')
                if sysid:
                    buf.append(' "%s"')
                buf.append('>\n')
                yield (Markup(''.join(buf)) % tuple([p for p in data if p]))
                have_doctype = True
            elif kind is XML_DECL and (not have_decl) and (not drop_xml_decl):
                version, encoding, standalone = data
                buf = ['<?xml version="%s"' % version]
                if encoding:
                    buf.append(' encoding="%s"' % encoding)
                if standalone != -1:
                    standalone = standalone and 'yes' or 'no'
                    buf.append(' standalone="%s"' % standalone)
                buf.append('?>\n')
                yield Markup(''.join(buf))
                have_decl = True
            elif kind is START_CDATA:
                yield Markup('<![CDATA[')
                in_cdata = True
            elif kind is END_CDATA:
                yield Markup(']]>')
                in_cdata = False
            elif kind is PI:
                yield _emit(kind, data, Markup('<?%s %s?>' % data))