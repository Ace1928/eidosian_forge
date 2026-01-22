import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class InvalidUriError(Exception):
    """Exception raised when URI is invalid."""

    def __init__(self, message):
        super(InvalidUriError, self).__init__(message)
        self.message = message