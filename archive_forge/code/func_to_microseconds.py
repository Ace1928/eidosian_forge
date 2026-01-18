import calendar
import datetime
import re
from cloudsdk.google.protobuf import timestamp_pb2
from_rfc3339_nanos = from_rfc3339  # from_rfc3339_nanos method was deprecated.
def to_microseconds(value):
    """Convert a datetime to microseconds since the unix epoch.

    Args:
        value (datetime.datetime): The datetime to covert.

    Returns:
        int: Microseconds since the unix epoch.
    """
    if not value.tzinfo:
        value = value.replace(tzinfo=datetime.timezone.utc)
    value = value.astimezone(datetime.timezone.utc)
    return int(calendar.timegm(value.timetuple()) * 1000000.0) + value.microsecond