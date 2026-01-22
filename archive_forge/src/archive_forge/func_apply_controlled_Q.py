from copy import copy
from typing import Sequence, Callable
import pennylane as qml
from pennylane import PauliX, Hadamard, MultiControlledX, CZ, adjoint
from pennylane.wires import Wires
from pennylane.templates import QFT
from pennylane.transforms.core import transform
@transform
def apply_controlled_Q(tape: qml.tape.QuantumTape, wires, target_wire, control_wire, work_wires) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Applies the transform that performs a controlled version of the :math:`\\mathcal{Q}` unitary
    defined in `this <https://arxiv.org/abs/1805.00109>`__ paper.

    The input ``tape`` should be the quantum circuit corresponding to the :math:`\\mathcal{F}` unitary
    in the paper above. This function transforms this circuit into a controlled version of the
    :math:`\\mathcal{Q}` unitary, which forms part of the quantum Monte Carlo algorithm. The
    :math:`\\mathcal{Q}` unitary encodes the target expectation value as a phase in one of its
    eigenvalues. This phase can be estimated using quantum phase estimation (see
    :class:`~.QuantumPhaseEstimation` for more details).

    Args:
        tape (QNode or QuantumTape or Callable): the quantum circuit that applies quantum operations
            according to the :math:`\\mathcal{F}` unitary used as part of quantum Monte Carlo estimation
        wires (Union[Wires or Sequence[int]]): the wires acted upon by the ``fn`` circuit
        target_wire (Union[Wires, int]): The wire in which the expectation value is encoded. Must be
            contained within ``wires``.
        control_wire (Union[Wires, int]): the control wire from the register of phase estimation
            qubits
        work_wires (Union[Wires, Sequence[int], or int]): additional work wires used when
            decomposing :math:`\\mathcal{Q}`

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will perform control on :math:`\\mathcal{Q}` unitary.

    Raises:
        ValueError: if ``target_wire`` is not in ``wires``
    """
    operations = tape.operations.copy()
    updated_operations = []
    with qml.queuing.QueuingManager.stop_recording():
        op_inv = [adjoint(copy(op)) for op in reversed(operations)]
        wires = Wires(wires)
        target_wire = Wires(target_wire)
        control_wire = Wires(control_wire)
        work_wires = Wires(work_wires) if work_wires is not None else Wires([])
        if not wires.contains_wires(target_wire):
            raise ValueError('The target wire must be contained within wires')
        updated_operations.extend(_apply_controlled_v(target_wire=target_wire, control_wire=control_wire))
        updated_operations.extend(op_inv)
        updated_operations.extend(_apply_controlled_z(wires=wires, control_wire=control_wire, work_wires=work_wires))
        updated_operations.extend(operations)
        updated_operations.extend(_apply_controlled_v(target_wire=target_wire, control_wire=control_wire))
        updated_operations.extend(op_inv)
        updated_operations.extend(_apply_controlled_z(wires=wires, control_wire=control_wire, work_wires=work_wires))
        updated_operations.extend(operations)
    tape = type(tape)(updated_operations, tape.measurements, shots=tape.shots)
    return ([tape], lambda x: x[0])