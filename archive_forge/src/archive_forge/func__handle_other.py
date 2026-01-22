from itertools import chain
import codecs
from xml.parsers import expat
import six
from six.moves import html_entities as entities, html_parser as html
from genshi.core import Attrs, QName, Stream, stripentities
from genshi.core import START, END, XML_DECL, DOCTYPE, TEXT, START_NS, \
from genshi.compat import StringIO, BytesIO
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