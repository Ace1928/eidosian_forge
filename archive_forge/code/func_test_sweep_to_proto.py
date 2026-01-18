import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
@pytest.mark.parametrize('sweep,expected', [(cirq.UnitSweep, cirq.UnitSweep), (cirq.Linspace('a', 0, 10, 25), cirq.Product(cirq.Zip(cirq.Linspace('a', 0, 10, 25)))), (cirq.Points('a', [1, 2, 3]), cirq.Product(cirq.Zip(cirq.Points('a', [1, 2, 3])))), (cirq.Zip(cirq.Linspace('a', 0, 1, 5), cirq.Points('b', [1, 2, 3])), cirq.Product(cirq.Zip(cirq.Linspace('a', 0, 1, 5), cirq.Points('b', [1, 2, 3])))), (cirq.Product(cirq.Linspace('a', 0, 1, 5), cirq.Points('b', [1, 2, 3])), cirq.Product(cirq.Zip(cirq.Linspace('a', 0, 1, 5)), cirq.Zip(cirq.Points('b', [1, 2, 3])))), (cirq.Product(cirq.Zip(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6])), cirq.Linspace('c', 0, 1, 5)), cirq.Product(cirq.Zip(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6])), cirq.Zip(cirq.Linspace('c', 0, 1, 5)))), (cirq.Product(cirq.Zip(cirq.Linspace('a', 0, 1, 5), cirq.Points('b', [1, 2, 3])), cirq.Zip(cirq.Linspace('c', 0, 1, 8), cirq.Points('d', [1, 0.5, 0.25, 0.125]))), cirq.Product(cirq.Zip(cirq.Linspace('a', 0, 1, 5), cirq.Points('b', [1, 2, 3])), cirq.Zip(cirq.Linspace('c', 0, 1, 8), cirq.Points('d', [1, 0.5, 0.25, 0.125]))))])
def test_sweep_to_proto(sweep, expected):
    proto = params.sweep_to_proto(sweep)
    out = params.sweep_from_proto(proto)
    assert out == expected