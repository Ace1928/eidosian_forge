import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class StorageCopyError(BotoServerError):
    """
    Error copying a key on a storage service.
    """
    pass