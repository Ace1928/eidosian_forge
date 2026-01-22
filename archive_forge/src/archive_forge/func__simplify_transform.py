from copy import copy
from typing import Callable, Union, Sequence
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.workflow import QNode
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumTape
@qml.transform
def _simplify_transform(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    with qml.QueuingManager.stop_recording():
        new_operations = [op.simplify() for op in tape.operations]
        new_measurements = [m.simplify() for m in tape.measurements]
    new_tape = type(tape)(new_operations, new_measurements, shots=tape.shots)

    def null_processing_fn(res):
        return res[0]
    return ([new_tape], null_processing_fn)