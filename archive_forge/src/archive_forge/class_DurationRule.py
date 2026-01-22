from datetime import datetime
from datetime import timedelta
from datetime import timezone
from cloudsdk.google.protobuf import duration_pb2
from cloudsdk.google.protobuf import timestamp_pb2
from proto import datetime_helpers, utils
class DurationRule:
    """A marshal between Python timedeltas and protobuf durations.

    Note: Python timedeltas are less precise than protobuf durations
    (microsecond vs. nanosecond level precision). If nanosecond-level
    precision matters, it is recommended to interact with the internal
    proto directly.
    """

    def to_python(self, value, *, absent: bool=None) -> timedelta:
        if isinstance(value, duration_pb2.Duration):
            return timedelta(days=value.seconds // 86400, seconds=value.seconds % 86400, microseconds=value.nanos // 1000)
        return value

    def to_proto(self, value) -> duration_pb2.Duration:
        if isinstance(value, timedelta):
            return duration_pb2.Duration(seconds=value.days * 86400 + value.seconds, nanos=value.microseconds * 1000)
        if isinstance(value, str):
            duration_value = duration_pb2.Duration()
            duration_value.FromJsonString(value=value)
            return duration_value
        return value