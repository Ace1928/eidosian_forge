import numpy as np
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
def test_results_from_proto_duplicate_qubit():
    measurements = [v2.MeasureInfo('foo', [q(0, 0), q(0, 1), q(1, 1)], instances=1, invert_mask=[False, False, False], tags=[])]
    proto = v2.result_pb2.Result()
    sr = proto.sweep_results.add()
    sr.repetitions = 8
    pr = sr.parameterized_results.add()
    pr.params.assignments.update({'i': 0})
    mr = pr.measurement_results.add()
    mr.key = 'foo'
    for qubit, results in [(q(0, 0), 204), (q(0, 1), 170), (q(0, 1), 240)]:
        qmr = mr.qubit_measurement_results.add()
        qmr.qubit.id = v2.qubit_to_proto_id(qubit)
        qmr.results = bytes([results])
    with pytest.raises(ValueError, match='Qubit already exists'):
        v2.results_from_proto(proto, measurements)