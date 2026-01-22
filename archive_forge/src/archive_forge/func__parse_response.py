from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
def _parse_response(self, parser, contenttype, body):
    if not contenttype.startswith('text/xml'):
        return body
    handler = XmlHandler(parser, self)
    xml.sax.parseString(body, handler)
    return parser