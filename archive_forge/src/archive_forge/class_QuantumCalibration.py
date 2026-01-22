import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class QuantumCalibration(proto.Message):
    """-

    Attributes:
        name (str):
            -
        timestamp (google.protobuf.timestamp_pb2.Timestamp):
            -
        data (google.protobuf.any_pb2.Any):
            -
    """
    name = proto.Field(proto.STRING, number=1)
    timestamp = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    data = proto.Field(proto.MESSAGE, number=3, message=any_pb2.Any)