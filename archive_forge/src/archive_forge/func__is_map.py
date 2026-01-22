from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
def _is_map(self, value):
    if isinstance(value, collections_abc.Mapping):
        return True
    return False