from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
def _serialize_ns(self, value):
    return [self._serialize_n(n) for n in value]