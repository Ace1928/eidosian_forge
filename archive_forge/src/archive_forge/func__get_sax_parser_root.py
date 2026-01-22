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
def _get_sax_parser_root(xml):
    """
    This function does some validation and rule check of xmlstring
    :param xml: string or object to be used in parsing reply
    :return: lxml object
    """
    if isinstance(xml, etree._Element):
        root = xml
    else:
        root = etree.fromstring(xml)
    return root