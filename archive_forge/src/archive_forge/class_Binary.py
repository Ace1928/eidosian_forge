import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
class Binary(bytes):

    def encode(self):
        return base64.b64encode(self).decode('utf-8')

    @property
    def value(self):
        return bytes(self)

    def __repr__(self):
        return 'Binary(%r)' % self.value