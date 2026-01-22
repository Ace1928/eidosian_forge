import base64
import boto
from boto.compat import StringIO
from boto.compat import six
from boto.sqs.attributes import Attributes
from boto.sqs.messageattributes import MessageAttributes
from boto.exception import SQSDecodeError
class EncodedMHMessage(MHMessage):
    """
    The EncodedMHMessage class provides a message that provides RFC821-like
    headers like this:

    HeaderName: HeaderValue

    This variation encodes/decodes the body of the message in base64 automatically.
    The message instance can be treated like a mapping object,
    i.e. m['HeaderName'] would return 'HeaderValue'.
    """

    def decode(self, value):
        try:
            value = base64.b64decode(value.encode('utf-8')).decode('utf-8')
        except:
            raise SQSDecodeError('Unable to decode message', self)
        return super(EncodedMHMessage, self).decode(value)

    def encode(self, value):
        value = super(EncodedMHMessage, self).encode(value)
        return base64.b64encode(value.encode('utf-8')).decode('utf-8')