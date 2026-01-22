import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class CreateQuantumProgramAndJobRequest(proto.Message):
    """-

    Attributes:
        parent (str):
            -
        quantum_program (google.cloud.quantum_v1alpha1.types.QuantumProgram):
            -
        quantum_job (google.cloud.quantum_v1alpha1.types.QuantumJob):
            -
    """
    parent = proto.Field(proto.STRING, number=1)
    quantum_program = proto.Field(proto.MESSAGE, number=2, message=quantum.QuantumProgram)
    quantum_job = proto.Field(proto.MESSAGE, number=3, message=quantum.QuantumJob)