import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
@pytest.mark.parametrize('bad_sweep', [cirq.Zip(cirq.Product(cirq.Linspace('a', 0, 10, 25), cirq.Linspace('b', 0, 10, 25))), cirq.Product(cirq.Zip(MySweep(key='a')))])
def test_sweep_to_proto_fail(bad_sweep):
    with pytest.raises(ValueError):
        params.sweep_to_proto(bad_sweep)