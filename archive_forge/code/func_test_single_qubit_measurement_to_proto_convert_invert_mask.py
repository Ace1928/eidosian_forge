import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_single_qubit_measurement_to_proto_convert_invert_mask():
    gate = cirq.MeasurementGate(1, 'test', invert_mask=(True,))
    proto = operations_pb2.Operation(measurement=operations_pb2.Measurement(targets=[operations_pb2.Qubit(row=2, col=3)], key='test', invert_mask=[True]))
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))