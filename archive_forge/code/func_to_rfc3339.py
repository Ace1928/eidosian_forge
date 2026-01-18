import calendar
import datetime
import re
from cloudsdk.google.protobuf import timestamp_pb2
from_rfc3339_nanos = from_rfc3339  # from_rfc3339_nanos method was deprecated.
def to_rfc3339(value, ignore_zone=True):
    """Convert a datetime to an RFC3339 timestamp string.

    Args:
        value (datetime.datetime):
            The datetime object to be converted to a string.
        ignore_zone (bool): If True, then the timezone (if any) of the
            datetime object is ignored and the datetime is treated as UTC.

    Returns:
        str: The RFC3339 formatted string representing the datetime.
    """
    if not ignore_zone and value.tzinfo is not None:
        value = value.replace(tzinfo=None) - value.utcoffset()
    return value.strftime(_RFC3339_MICROS)