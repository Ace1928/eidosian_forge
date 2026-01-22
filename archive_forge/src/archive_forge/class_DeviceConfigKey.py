import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class DeviceConfigKey(proto.Message):
    """-
    Attributes:
        run_name (str):
            -
        config_alias (str):
            -
    """
    run_name = proto.Field(proto.STRING, number=1)
    config_alias = proto.Field(proto.STRING, number=2)