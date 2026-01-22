import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class Timing(proto.Message):
    """-

        Attributes:
            started_time (google.protobuf.timestamp_pb2.Timestamp):
                -
            completed_time (google.protobuf.timestamp_pb2.Timestamp):
                -
        """
    started_time = proto.Field(proto.MESSAGE, number=1, message=timestamp_pb2.Timestamp)
    completed_time = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)