import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class TooManyRecordsException(Exception):
    """
    Exception raised when a search of Route53 records returns more
    records than requested.
    """

    def __init__(self, message):
        super(TooManyRecordsException, self).__init__(message)
        self.message = message