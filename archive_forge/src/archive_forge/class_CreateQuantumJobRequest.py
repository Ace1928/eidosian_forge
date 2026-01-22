import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class CreateQuantumJobRequest(proto.Message):
    """-

    Attributes:
        parent (str):
            -
        quantum_job (google.cloud.quantum_v1alpha1.types.QuantumJob):
            -
        overwrite_existing_run_context (bool):
            -
    """
    parent = proto.Field(proto.STRING, number=1)
    quantum_job = proto.Field(proto.MESSAGE, number=2, message=quantum.QuantumJob)
    overwrite_existing_run_context = proto.Field(proto.BOOL, number=3)