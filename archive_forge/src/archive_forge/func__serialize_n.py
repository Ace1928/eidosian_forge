from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
def _serialize_n(self, value):
    number = str(DYNAMODB_CONTEXT.create_decimal(value))
    if number in ['Infinity', 'NaN']:
        raise TypeError('Infinity and NaN not supported')
    return number