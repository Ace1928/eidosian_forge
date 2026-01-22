import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class ActivityStats(proto.Message):
    """-

        Attributes:
            active_users_count (int):
                -
            active_jobs_count (int):
                -
        """
    active_users_count = proto.Field(proto.INT64, number=1)
    active_jobs_count = proto.Field(proto.INT64, number=2)