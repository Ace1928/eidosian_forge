import contextlib
import copy
from collections import Counter
from typing import List, Union, Optional, Sequence
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import TensorLike
from pennylane.operation import Observable, Operator, Operation, _UNSET_BATCH_SIZE
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, process_queue
from pennylane.wires import Wires
def to_openqasm(self, wires=None, rotations=True, measure_all=True, precision=None):
    """Serialize the circuit as an OpenQASM 2.0 program.

        Measurements are assumed to be performed on all qubits in the computational basis. An
        optional ``rotations`` argument can be provided so that output of the OpenQASM circuit is
        diagonal in the eigenbasis of the quantum script's observables. The measurement outputs can be
        restricted to only those specified in the script by setting ``measure_all=False``.

        .. note::

            The serialized OpenQASM program assumes that gate definitions
            in ``qelib1.inc`` are available.

        Args:
            wires (Wires or None): the wires to use when serializing the circuit
            rotations (bool): in addition to serializing user-specified
                operations, also include the gates that diagonalize the
                measured wires such that they are in the eigenbasis of the circuit observables.
            measure_all (bool): whether to perform a computational basis measurement on all qubits
                or just those specified in the script
            precision (int): decimal digits to display for parameters

        Returns:
            str: OpenQASM serialization of the circuit
        """
    wires = wires or self.wires
    qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    if self.num_wires == 0:
        return qasm_str
    qasm_str += f'qreg q[{len(wires)}];\n'
    qasm_str += f'creg c[{len(wires)}];\n'
    operations = qml.transforms.convert_to_numpy_parameters(self).operations
    if rotations:
        operations += self.diagonalizing_gates
    just_ops = QuantumScript(operations)
    operations = just_ops.expand(depth=10, stop_at=lambda obj: obj.name in OPENQASM_GATES).operations
    for op in operations:
        try:
            gate = OPENQASM_GATES[op.name]
        except KeyError as e:
            raise ValueError(f'Operation {op.name} not supported by the QASM serializer') from e
        wire_labels = ','.join([f'q[{wires.index(w)}]' for w in op.wires.tolist()])
        params = ''
        if op.num_params > 0:
            if precision is not None:
                params = '(' + ','.join([f'{p:.{precision}}' for p in op.parameters]) + ')'
            else:
                params = '(' + ','.join([str(p) for p in op.parameters]) + ')'
        qasm_str += f'{gate}{params} {wire_labels};\n'
    if measure_all:
        for wire in range(len(wires)):
            qasm_str += f'measure q[{wire}] -> c[{wire}];\n'
    else:
        measured_wires = Wires.all_wires([m.wires for m in self.measurements])
        for w in measured_wires:
            wire_indx = self.wires.index(w)
            qasm_str += f'measure q[{wire_indx}] -> c[{wire_indx}];\n'
    return qasm_str