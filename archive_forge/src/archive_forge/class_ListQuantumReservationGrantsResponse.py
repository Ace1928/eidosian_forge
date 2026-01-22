import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class ListQuantumReservationGrantsResponse(proto.Message):
    """-

    Attributes:
        reservation_grants (Sequence[google.cloud.quantum_v1alpha1.types.QuantumReservationGrant]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self
    reservation_grants = proto.RepeatedField(proto.MESSAGE, number=1, message=quantum.QuantumReservationGrant)
    next_page_token = proto.Field(proto.STRING, number=2)