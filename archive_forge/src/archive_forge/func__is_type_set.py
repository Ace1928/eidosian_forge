from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
def _is_type_set(self, value, type_validator):
    if self._is_set(value):
        if False not in map(type_validator, value):
            return True
    return False