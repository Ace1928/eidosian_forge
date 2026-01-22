import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class GSCreateError(StorageCreateError):
    """
    Error creating a bucket or key on GS.
    """
    pass