from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
def _is_set(self, value):
    if isinstance(value, collections_abc.Set):
        return True
    return False