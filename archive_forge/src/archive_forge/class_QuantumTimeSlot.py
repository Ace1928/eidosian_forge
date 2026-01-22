import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class QuantumTimeSlot(proto.Message):
    """-

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        processor_name (str):
            -
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        time_slot_type (google.cloud.quantum_v1alpha1.types.QuantumTimeSlot.TimeSlotType):
            -
        reservation_config (google.cloud.quantum_v1alpha1.types.QuantumTimeSlot.ReservationConfig):
            -

            This field is a member of `oneof`_ ``type_config``.
        maintenance_config (google.cloud.quantum_v1alpha1.types.QuantumTimeSlot.MaintenanceConfig):
            -

            This field is a member of `oneof`_ ``type_config``.
    """

    class TimeSlotType(proto.Enum):
        """-"""
        TIME_SLOT_TYPE_UNSPECIFIED = 0
        MAINTENANCE = 1
        OPEN_SWIM = 2
        RESERVATION = 3
        UNALLOCATED = 4

    class ReservationConfig(proto.Message):
        """-

        Attributes:
            reservation (str):
                -
            project_id (str):
                -
            whitelisted_users (Sequence[str]):
                -
        """
        reservation = proto.Field(proto.STRING, number=3)
        project_id = proto.Field(proto.STRING, number=1)
        whitelisted_users = proto.RepeatedField(proto.STRING, number=2)

    class MaintenanceConfig(proto.Message):
        """-

        Attributes:
            title (str):
                -
            description (str):
                -
        """
        title = proto.Field(proto.STRING, number=1)
        description = proto.Field(proto.STRING, number=2)
    processor_name = proto.Field(proto.STRING, number=1)
    start_time = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    end_time = proto.Field(proto.MESSAGE, number=3, message=timestamp_pb2.Timestamp)
    time_slot_type = proto.Field(proto.ENUM, number=5, enum=TimeSlotType)
    reservation_config = proto.Field(proto.MESSAGE, number=6, oneof='type_config', message=ReservationConfig)
    maintenance_config = proto.Field(proto.MESSAGE, number=7, oneof='type_config', message=MaintenanceConfig)