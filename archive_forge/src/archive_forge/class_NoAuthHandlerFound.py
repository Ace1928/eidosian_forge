import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class NoAuthHandlerFound(Exception):
    """Is raised when no auth handlers were found ready to authenticate."""
    pass