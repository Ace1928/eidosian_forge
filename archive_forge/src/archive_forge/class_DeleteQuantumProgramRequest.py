import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class DeleteQuantumProgramRequest(proto.Message):
    """-

    Attributes:
        name (str):
            -
        delete_jobs (bool):
            -
    """
    name = proto.Field(proto.STRING, number=1)
    delete_jobs = proto.Field(proto.BOOL, number=2)