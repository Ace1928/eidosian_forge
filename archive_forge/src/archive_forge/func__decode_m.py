import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
def _decode_m(self, attr):
    return dict([(k, self.decode(v)) for k, v in attr.items()])