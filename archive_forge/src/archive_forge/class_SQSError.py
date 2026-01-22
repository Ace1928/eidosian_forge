import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class SQSError(BotoServerError):
    """
    General Error on Simple Queue Service.
    """

    def __init__(self, status, reason, body=None):
        self.detail = None
        self.type = None
        super(SQSError, self).__init__(status, reason, body)

    def startElement(self, name, attrs, connection):
        return super(SQSError, self).startElement(name, attrs, connection)

    def endElement(self, name, value, connection):
        if name == 'Detail':
            self.detail = value
        elif name == 'Type':
            self.type = value
        else:
            return super(SQSError, self).endElement(name, value, connection)

    def _cleanupParsedProperties(self):
        super(SQSError, self)._cleanupParsedProperties()
        for p in ('detail', 'type'):
            setattr(self, p, None)