from functools import partial
from typing import Callable, Sequence
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import DefaultQubit, DefaultQubitLegacy, DefaultMixed
from pennylane.measurements import StateMP, DensityMatrixMP
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane import transform
@partial(transform, final_transform=True)
def vn_entropy(tape: QuantumTape, wires: Sequence[int], base: float=None, **kwargs) -> (Sequence[QuantumTape], Callable):
    """Compute the Von Neumann entropy from a :class:`.QuantumTape` returning a :func:`~pennylane.state`.

    .. math::
        S( \\rho ) = -\\text{Tr}( \\rho \\log ( \\rho ))

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit returning a :func:`~pennylane.state`.
        wires (Sequence(int)): List of wires in the considered subsystem.
        base (float): Base for the logarithm, default is None the natural logarithm is used in this case.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the Von Neumann entropy in the form of a tensor.

    **Example**

    It is possible to obtain the entropy of a subsystem from a :class:`.QNode` returning a :func:`~pennylane.state`.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)
        @qml.qnode(dev)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

    >>> vn_entropy(circuit, wires=[0])(np.pi/2)
    0.6931471805599453

    The function is differentiable with backpropagation for all interfaces, e.g.:

    >>> param = np.array(np.pi/4, requires_grad=True)
    >>> qml.grad(vn_entropy(circuit, wires=[0]))(param)
    tensor(0.62322524, requires_grad=True)

    .. seealso:: :func:`pennylane.math.vn_entropy` and :func:`pennylane.vn_entropy`
    """
    all_wires = kwargs.get('device_wires', tape.wires)
    wire_map = {w: i for i, w in enumerate(all_wires)}
    indices = [wire_map[w] for w in wires]
    measurements = tape.measurements
    if len(measurements) != 1 or not isinstance(measurements[0], StateMP):
        raise ValueError('The qfunc return type needs to be a state.')

    def processing_fn(res):
        device = kwargs.get('device', None)
        c_dtype = getattr(device, 'C_DTYPE', 'complex128')
        if not isinstance(measurements[0], DensityMatrixMP) and (not isinstance(device, DefaultMixed)):
            if len(wires) == len(all_wires):
                return 0.0
            density_matrix = qml.math.dm_from_state_vector(res[0], c_dtype=c_dtype)
            entropy = qml.math.vn_entropy(density_matrix, indices, base, c_dtype=c_dtype)
            return entropy
        entropy = qml.math.vn_entropy(res[0], indices, base, c_dtype)
        return entropy
    return ([tape], processing_fn)