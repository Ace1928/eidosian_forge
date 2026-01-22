import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class SQSDecodeError(BotoClientError):
    """
    Error when decoding an SQS message.
    """

    def __init__(self, reason, message):
        super(SQSDecodeError, self).__init__(reason, message)
        self.message = message

    def __repr__(self):
        return 'SQSDecodeError: %s' % self.reason

    def __str__(self):
        return 'SQSDecodeError: %s' % self.reason