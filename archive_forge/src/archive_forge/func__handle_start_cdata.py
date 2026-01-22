from itertools import chain
import codecs
from xml.parsers import expat
import six
from six.moves import html_entities as entities, html_parser as html
from genshi.core import Attrs, QName, Stream, stripentities
from genshi.core import START, END, XML_DECL, DOCTYPE, TEXT, START_NS, \
from genshi.compat import StringIO, BytesIO
def _handle_start_cdata(self):
    self._enqueue(START_CDATA)