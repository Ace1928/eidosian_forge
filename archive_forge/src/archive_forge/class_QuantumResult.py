import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class QuantumResult(proto.Message):
    """-

    Attributes:
        parent (str):
            -
        result (google.protobuf.any_pb2.Any):
            -
    """
    parent = proto.Field(proto.STRING, number=1)
    result = proto.Field(proto.MESSAGE, number=2, message=any_pb2.Any)