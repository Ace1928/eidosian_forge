import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class ListQuantumReservationsResponse(proto.Message):
    """-

    Attributes:
        reservations (Sequence[google.cloud.quantum_v1alpha1.types.QuantumReservation]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self
    reservations = proto.RepeatedField(proto.MESSAGE, number=1, message=quantum.QuantumReservation)
    next_page_token = proto.Field(proto.STRING, number=2)