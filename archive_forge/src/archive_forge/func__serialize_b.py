from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
def _serialize_b(self, value):
    if isinstance(value, Binary):
        value = value.value
    return value