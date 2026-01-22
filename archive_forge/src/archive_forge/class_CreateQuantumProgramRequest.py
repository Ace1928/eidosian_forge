import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class CreateQuantumProgramRequest(proto.Message):
    """-

    Attributes:
        parent (str):
            -
        quantum_program (google.cloud.quantum_v1alpha1.types.QuantumProgram):
            -
        overwrite_existing_source_code (bool):
            -
    """
    parent = proto.Field(proto.STRING, number=1)
    quantum_program = proto.Field(proto.MESSAGE, number=2, message=quantum.QuantumProgram)
    overwrite_existing_source_code = proto.Field(proto.BOOL, number=3)