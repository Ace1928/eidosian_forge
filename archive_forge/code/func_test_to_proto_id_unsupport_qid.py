import pytest
import cirq
import cirq_google.api.v2 as v2
def test_to_proto_id_unsupport_qid():

    class ValidQubit(cirq.Qid):

        def __init__(self, name):
            self._name = name

        @property
        def dimension(self):
            pass

        def _comparison_key(self):
            pass
    with pytest.raises(ValueError, match='ValidQubit'):
        _ = v2.qubit_to_proto_id(ValidQubit('d'))