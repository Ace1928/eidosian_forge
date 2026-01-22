import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
class MottonenStatePreparation(Operation):
    """
    Prepares an arbitrary state on the given wires using a decomposition into gates developed
    by `Möttönen et al. (2004) <https://arxiv.org/abs/quant-ph/0407010>`_.

    The state is prepared via a sequence
    of uniformly controlled rotations. A uniformly controlled rotation on a target qubit is
    composed from all possible controlled rotations on the qubit and can be used to address individual
    elements of the state vector.

    In the work of Möttönen et al., inverse state preparation
    is executed by first equalizing the phases of the state vector via uniformly controlled Z rotations,
    and then rotating the now real state vector into the direction of the state :math:`|0\\rangle` via
    uniformly controlled Y rotations.

    This code is adapted from code written by Carsten Blank for PennyLane-Qiskit.

    .. warning::

        Due to non-trivial classical processing of the state vector,
        this template is not always fully differentiable.

    Args:
        state_vector (tensor_like): Input array of shape ``(2^n,)``, where ``n`` is the number of wires
            the state preparation acts on. The input array must be normalized.
        wires (Iterable): wires that the template acts on

    Example:

        ``MottonenStatePreparation`` creates any arbitrary state on the given wires depending on the input state vector.

        .. code-block:: python

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(state):
                qml.MottonenStatePreparation(state_vector=state, wires=range(3))
                return qml.state()

            state = np.array([1, 2j, 3, 4j, 5, 6j, 7, 8j])
            state = state / np.linalg.norm(state)

            print(qml.draw(circuit, expansion_strategy="device", max_length=80)(state))

        .. code-block::

            0: ──RY(2.35)─╭●───────────╭●──────────────╭●────────────────────────╭●
            1: ──RY(2.09)─╰X──RY(0.21)─╰X─╭●───────────│────────────╭●───────────│─
            2: ──RY(1.88)─────────────────╰X──RY(0.10)─╰X──RY(0.08)─╰X──RY(0.15)─╰X

            ──╭●────────╭●────╭●────╭●─╭GlobalPhase(-0.79)─┤ ╭State
            ──╰X────────╰X─╭●─│──╭●─│──├GlobalPhase(-0.79)─┤ ├State
            ───RZ(1.57)────╰X─╰X─╰X─╰X─╰GlobalPhase(-0.79)─┤ ╰State

        The state preparation can be checked by running:

        >>> print(np.allclose(state, circuit(state)))
        True

    """
    num_wires = AnyWires
    grad_method = None
    ndim_params = (1,)

    def __init__(self, state_vector, wires, id=None):
        batched = len(qml.math.shape(state_vector)) > 1
        state_batch = state_vector if batched else [state_vector]
        for i, state in enumerate(state_batch):
            shape = qml.math.shape(state)
            if len(shape) != 1:
                raise ValueError(f'State vectors must be one-dimensional; vector {i} has shape {shape}.')
            n_amplitudes = shape[0]
            if n_amplitudes != 2 ** len(qml.wires.Wires(wires)):
                raise ValueError(f'State vectors must be of length {2 ** len(wires)} or less; vector {i} has length {n_amplitudes}.')
            if not qml.math.is_abstract(state):
                norm = qml.math.sum(qml.math.abs(state) ** 2)
                if not qml.math.allclose(norm, 1.0, atol=0.001):
                    raise ValueError(f'State vectors have to be of norm 1.0, vector {i} has norm {norm}')
        super().__init__(state_vector, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(state_vector, wires):
        """Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \\dots O_n.



        .. seealso:: :meth:`~.MottonenStatePreparation.decomposition`.

        Args:
            state_vector (tensor_like): Normalized state vector of shape ``(2^len(wires),)``
            wires (Any or Iterable[Any]): wires that the operator acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> state_vector = torch.tensor([0.5, 0.5, 0.5, 0.5])
        >>> qml.MottonenStatePreparation.compute_decomposition(state_vector, wires=["a", "b"])
        [RY(array(1.57079633), wires=['a']),
        RY(array(1.57079633), wires=['b']),
        CNOT(wires=['a', 'b']),
        CNOT(wires=['a', 'b'])]
        """
        if len(qml.math.shape(state_vector)) > 1:
            raise ValueError('Broadcasting with MottonenStatePreparation is not supported. Please use the qml.transforms.broadcast_expand transform to use broadcasting with MottonenStatePreparation.')
        a = qml.math.abs(state_vector)
        omega = qml.math.angle(state_vector)
        wires_reverse = wires[::-1]
        op_list = []
        for k in range(len(wires_reverse), 0, -1):
            alpha_y_k = _get_alpha_y(a, len(wires_reverse), k)
            control = wires_reverse[k:]
            target = wires_reverse[k - 1]
            op_list.extend(_apply_uniform_rotation_dagger(qml.RY, alpha_y_k, control, target))
        if qml.math.is_abstract(omega) or not qml.math.allclose(omega, 0):
            for k in range(len(wires_reverse), 0, -1):
                alpha_z_k = _get_alpha_z(omega, len(wires_reverse), k)
                control = wires_reverse[k:]
                target = wires_reverse[k - 1]
                if len(alpha_z_k) > 0:
                    op_list.extend(_apply_uniform_rotation_dagger(qml.RZ, alpha_z_k, control, target))
            global_phase = qml.math.sum(-1 * qml.math.angle(state_vector) / len(state_vector))
            op_list.extend([qml.GlobalPhase(global_phase, wires=wires)])
        return op_list