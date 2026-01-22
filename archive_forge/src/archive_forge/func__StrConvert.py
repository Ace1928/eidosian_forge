import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def _StrConvert(value):
    """Converts value to str if it is not."""
    if not isinstance(value, str):
        return value.encode('utf-8')
    return value