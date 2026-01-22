import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class QuantumRunStreamRequest(proto.Message):
    """-

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        message_id (str):
            -
        parent (str):
            -
        create_quantum_program_and_job (google.cloud.quantum_v1alpha1.types.CreateQuantumProgramAndJobRequest):
            -

            This field is a member of `oneof`_ ``request``.
        create_quantum_job (google.cloud.quantum_v1alpha1.types.CreateQuantumJobRequest):
            -

            This field is a member of `oneof`_ ``request``.
        get_quantum_result (google.cloud.quantum_v1alpha1.types.GetQuantumResultRequest):
            -

            This field is a member of `oneof`_ ``request``.
    """
    message_id = proto.Field(proto.STRING, number=1)
    parent = proto.Field(proto.STRING, number=2)
    create_quantum_program_and_job = proto.Field(proto.MESSAGE, number=3, oneof='request', message='CreateQuantumProgramAndJobRequest')
    create_quantum_job = proto.Field(proto.MESSAGE, number=4, oneof='request', message='CreateQuantumJobRequest')
    get_quantum_result = proto.Field(proto.MESSAGE, number=5, oneof='request', message='GetQuantumResultRequest')