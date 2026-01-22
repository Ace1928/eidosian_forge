import pennylane as qml
from pennylane.operation import AnyWires, Operation
class LocalHilbertSchmidt(HilbertSchmidt):
    """Create a Local Hilbert-Schmidt template that can be used to compute the  Local Hilbert-Schmidt Test (LHST).
    The result of the LHST is a useful quantity for compiling a unitary ``U`` with an approximate unitary ``V``. The
    LHST is used as a distance between `U` and `V`, it is similar to the Hilbert-Schmidt test, but the measurement is
    made only on one qubit at the end of the circuit. The LHST cost is always smaller than the HST cost and is useful
    for large unitaries.

    .. figure:: ../../_static/templates/subroutines/lhst.png
        :align: center
        :width: 80%
        :target: javascript:void(0);

    Args:
        params (array): Parameters for the quantum function `V`.
        v_function (Callable): Quantum function that represents the approximate compiled unitary `V`.
        v_wires (int or Iterable[Number, str]]): the wire(s) the approximate compiled unitary act on.
        u_tape (.QuantumTape): `U`, the unitary to be compiled as a ``qml.tape.QuantumTape``.

    Raises:
        QuantumFunctionError: The argument u_tape must be a QuantumTape
        QuantumFunctionError: ``v_function`` is not a valid Quantum function.
        QuantumFunctionError: `U` and `V` do not have the same number of wires.
        QuantumFunctionError: The wires ``v_wires`` are a subset of `V` wires.
        QuantumFunctionError: u_tape and v_tape must act on distinct wires.

    **Reference**

    [1] Sumeet Khatri, Ryan LaRose, Alexander Poremba, Lukasz Cincio, Andrew T. Sornborger and Patrick J. Coles
    Quantum-assisted Quantum Compiling.
    `arxiv/1807.00800 <https://arxiv.org/pdf/1807.00800.pdf>`_

    .. seealso:: :class:`~.HilbertSchmidt`

    .. details::
        :title: Usage Details

        Consider that we want to evaluate the Local Hilbert-Schmidt Test cost between the unitary ``U`` and an
        approximate unitary ``V``. We need to define some functions where it is possible to use the
        :class:`~.LocalHilbertSchmidt` template. Here the considered unitary is ``CZ`` and we try to compute the
        cost for the approximate unitary.

        .. code-block:: python

            import numpy as np

            with qml.QueuingManager.stop_recording():
                u_tape = qml.tape.QuantumTape([qml.CZ(wires=(0,1))])

            def v_function(params):
                qml.RZ(params[0], wires=2)
                qml.RZ(params[1], wires=3)
                qml.CNOT(wires=[2, 3])
                qml.RZ(params[2], wires=3)
                qml.CNOT(wires=[2, 3])

            dev = qml.device("default.qubit", wires=4)

            @qml.qnode(dev)
            def local_hilbert_test(v_params, v_function, v_wires, u_tape):
                qml.LocalHilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
                return qml.probs(u_tape.wires + v_wires)

            def cost_lhst(parameters, v_function, v_wires, u_tape):
                return (1 - local_hilbert_test(v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape)[0])

        Now that the cost function has been defined it can be called for specific parameters:

        >>> cost_lhst([3*np.pi/2, 3*np.pi/2, np.pi/2], v_function = v_function, v_wires = [2,3], u_tape = u_tape)
        0.5
    """

    @staticmethod
    def compute_decomposition(params, wires, u_tape, v_tape, v_function=None, v_wires=None):
        """Representation of the operator as a product of other operators (static method)."""
        n_wires = len(u_tape.wires + v_tape.wires)
        first_range = range(n_wires // 2)
        second_range = range(n_wires // 2, n_wires)
        decomp_ops = [qml.Hadamard(wires[i]) for i in first_range]
        decomp_ops.extend((qml.CNOT(wires=[wires[i], wires[j]]) for i, j in zip(first_range, second_range)))
        if qml.QueuingManager.recording():
            decomp_ops.extend((qml.apply(op_u) for op_u in u_tape.operations))
        else:
            decomp_ops.extend(u_tape.operations)
        decomp_ops.extend((qml.adjoint(qml.apply, lazy=False)(op_v) for op_v in v_tape.operations))
        decomp_ops.extend((qml.CNOT(wires=[wires[0], wires[n_wires // 2]]), qml.Hadamard(wires[0])))
        return decomp_ops