import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class InvalidLifecycleConfigError(Exception):
    """Exception raised when GCS lifecycle configuration XML is invalid."""

    def __init__(self, message):
        super(InvalidLifecycleConfigError, self).__init__(message)
        self.message = message