import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class S3DataError(StorageDataError):
    """
    Error receiving data from S3.
    """
    pass