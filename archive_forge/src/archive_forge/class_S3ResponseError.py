import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class S3ResponseError(StorageResponseError):
    """
    Error in response from S3.
    """

    def __init__(self, status, reason, body=None):
        self.region = None
        self.endpoint = None
        super(StorageResponseError, self).__init__(status, reason, body)

    def startElement(self, name, attrs, connection):
        return super(StorageResponseError, self).startElement(name, attrs, connection)

    def endElement(self, name, value, connection):
        if name == 'Region':
            self.region = value
        elif name == 'LocationConstraint':
            self.region = value
        elif name == 'Endpoint':
            self.endpoint = value
        else:
            return super(StorageResponseError, self).endElement(name, value, connection)

    def _cleanupParsedProperties(self):
        super(StorageResponseError, self)._cleanupParsedProperties()
        for p in ('region', 'endpoint'):
            setattr(self, p, None)