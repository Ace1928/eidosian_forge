from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
def _serialize_m(self, value):
    return dict([(k, self.serialize(v)) for k, v in value.items()])