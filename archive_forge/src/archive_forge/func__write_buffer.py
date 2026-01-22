import os
from threading import Lock
import difflib
from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.builder import E
from xml.sax._exceptions import SAXParseException
from xml.sax import make_parser
from ncclient.transport.parser import DefaultXMLParser
from ncclient.operations import rpc
from ncclient.transport.parser import SAXFilterXMLNotFoundError
from ncclient.transport.parser import MSG_DELIM, MSG_DELIM_LEN
from ncclient.operations.errors import OperationError
import logging
from ncclient.xml_ import BASE_NS_1_0
def _write_buffer(self, content, format_str, **kwargs):
    self._session._buffer.seek(0, os.SEEK_END)
    attrs = ''
    for name, value in kwargs.items():
        attr = ' {}={}'.format(name, quoteattr(value))
        attrs = attrs + attr
    data = format_str.format(escape(content), attrs)
    self._session._buffer.write(str.encode(data))