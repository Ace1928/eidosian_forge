from itertools import chain
import codecs
from xml.parsers import expat
import six
from six.moves import html_entities as entities, html_parser as html
from genshi.core import Attrs, QName, Stream, stripentities
from genshi.core import START, END, XML_DECL, DOCTYPE, TEXT, START_NS, \
from genshi.compat import StringIO, BytesIO
def ET(element):
    """Convert a given ElementTree element to a markup stream.
    
    :param element: an ElementTree element
    :return: a markup stream
    """
    tag_name = QName(element.tag.lstrip('{'))
    attrs = Attrs([(QName(attr.lstrip('{')), value) for attr, value in element.items()])
    yield (START, (tag_name, attrs), (None, -1, -1))
    if element.text:
        yield (TEXT, element.text, (None, -1, -1))
    for child in element:
        for item in ET(child):
            yield item
    yield (END, tag_name, (None, -1, -1))
    if element.tail:
        yield (TEXT, element.tail, (None, -1, -1))