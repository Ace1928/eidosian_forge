import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_string_format():
    x, y, z = cirq.LineQubit.range(3)
    fc0 = cirq.FrozenCircuit()
    op0 = cirq.CircuitOperation(fc0)
    assert str(op0) == '[  ]'
    fc0_global_phase_inner = cirq.FrozenCircuit(cirq.global_phase_operation(1j), cirq.global_phase_operation(1j))
    op0_global_phase_inner = cirq.CircuitOperation(fc0_global_phase_inner)
    fc0_global_phase_outer = cirq.FrozenCircuit(op0_global_phase_inner, cirq.global_phase_operation(1j))
    op0_global_phase_outer = cirq.CircuitOperation(fc0_global_phase_outer)
    assert str(op0_global_phase_outer) == '[                       ]\n[                       ]\n[ global phase:   -0.5π ]'
    fc1 = cirq.FrozenCircuit(cirq.X(x), cirq.H(y), cirq.CX(y, z), cirq.measure(x, y, z, key='m'))
    op1 = cirq.CircuitOperation(fc1)
    assert str(op1) == "[ 0: ───X───────M('m')─── ]\n[               │         ]\n[ 1: ───H───@───M──────── ]\n[           │   │         ]\n[ 2: ───────X───M──────── ]"
    assert repr(op1) == "cirq.CircuitOperation(\n    circuit=cirq.FrozenCircuit([\n        cirq.Moment(\n            cirq.X(cirq.LineQubit(0)),\n            cirq.H(cirq.LineQubit(1)),\n        ),\n        cirq.Moment(\n            cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(2)),\n        ),\n        cirq.Moment(\n            cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2), key=cirq.MeasurementKey(name='m')),\n        ),\n    ]),\n)"
    fc2 = cirq.FrozenCircuit(cirq.X(x), cirq.H(y), cirq.CX(y, x))
    op2 = cirq.CircuitOperation(circuit=fc2, qubit_map={y: z}, repetitions=3, parent_path=('outer', 'inner'), repetition_ids=['a', 'b', 'c'])
    assert str(op2) == "[ 0: ───X───X─── ]\n[           │    ]\n[ 1: ───H───@─── ](qubit_map={q(1): q(2)}, parent_path=('outer', 'inner'), repetition_ids=['a', 'b', 'c'])"
    assert repr(op2) == "cirq.CircuitOperation(\n    circuit=cirq.FrozenCircuit([\n        cirq.Moment(\n            cirq.X(cirq.LineQubit(0)),\n            cirq.H(cirq.LineQubit(1)),\n        ),\n        cirq.Moment(\n            cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(0)),\n        ),\n    ]),\n    repetitions=3,\n    qubit_map={cirq.LineQubit(1): cirq.LineQubit(2)},\n    parent_path=('outer', 'inner'),\n    repetition_ids=['a', 'b', 'c'],\n)"
    fc3 = cirq.FrozenCircuit(cirq.X(x) ** sympy.Symbol('b'), cirq.measure(x, key='m'))
    op3 = cirq.CircuitOperation(circuit=fc3, qubit_map={x: y}, measurement_key_map={'m': 'p'}, param_resolver={sympy.Symbol('b'): 2})
    indented_fc3_repr = repr(fc3).replace('\n', '\n    ')
    assert str(op3) == "[ 0: ───X^b───M('m')─── ](qubit_map={q(0): q(1)}, key_map={m: p}, params={b: 2})"
    assert repr(op3) == f"cirq.CircuitOperation(\n    circuit={indented_fc3_repr},\n    qubit_map={{cirq.LineQubit(0): cirq.LineQubit(1)}},\n    measurement_key_map={{'m': 'p'}},\n    param_resolver=cirq.ParamResolver({{sympy.Symbol('b'): 2}}),\n)"
    fc4 = cirq.FrozenCircuit(cirq.X(y))
    op4 = cirq.CircuitOperation(fc4)
    fc5 = cirq.FrozenCircuit(cirq.X(x), op4)
    op5 = cirq.CircuitOperation(fc5)
    assert repr(op5) == 'cirq.CircuitOperation(\n    circuit=cirq.FrozenCircuit([\n        cirq.Moment(\n            cirq.X(cirq.LineQubit(0)),\n            cirq.CircuitOperation(\n                circuit=cirq.FrozenCircuit([\n                    cirq.Moment(\n                        cirq.X(cirq.LineQubit(1)),\n                    ),\n                ]),\n            ),\n        ),\n    ]),\n)'
    op6 = cirq.CircuitOperation(fc5, use_repetition_ids=False)
    assert repr(op6) == 'cirq.CircuitOperation(\n    circuit=cirq.FrozenCircuit([\n        cirq.Moment(\n            cirq.X(cirq.LineQubit(0)),\n            cirq.CircuitOperation(\n                circuit=cirq.FrozenCircuit([\n                    cirq.Moment(\n                        cirq.X(cirq.LineQubit(1)),\n                    ),\n                ]),\n            ),\n        ),\n    ]),\n    use_repetition_ids=False,\n)'
    op7 = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(x, key='a')), use_repetition_ids=False, repeat_until=cirq.KeyCondition(cirq.MeasurementKey('a')))
    assert repr(op7) == "cirq.CircuitOperation(\n    circuit=cirq.FrozenCircuit([\n        cirq.Moment(\n            cirq.measure(cirq.LineQubit(0), key=cirq.MeasurementKey(name='a')),\n        ),\n    ]),\n    use_repetition_ids=False,\n    repeat_until=cirq.KeyCondition(cirq.MeasurementKey(name='a')),\n)"