import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class StorageDataError(BotoClientError):
    """
    Error receiving data from a storage service.
    """
    pass