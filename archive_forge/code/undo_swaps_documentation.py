from typing import Sequence, Callable
from pennylane.transforms import transform
from pennylane.tape import QuantumTape
Quantum function transform to remove SWAP gates by running from right
    to left through the circuit changing the position of the qubits accordingly.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    >>> dev = qml.device('default.qubit', wires=3)

    You can apply the transform directly on a :class:`QNode`

    .. code-block:: python

        @undo_swaps
        @qml.qnode(device=dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.X(1)
            qml.SWAP(wires=[0,1])
            qml.SWAP(wires=[0,2])
            qml.Y(0)
            return qml.expval(qml.Z(0))

    The SWAP gates are removed before execution.

    .. details::
        :title: Usage Details

        Consider the following quantum function:

        .. code-block:: python

            def qfunc():
                qml.Hadamard(wires=0)
                qml.X(1)
                qml.SWAP(wires=[0,1])
                qml.SWAP(wires=[0,2])
                qml.Y(0)
                return qml.expval(qml.Z(0))

        The circuit before optimization:

        >>> dev = qml.device('default.qubit', wires=3)
        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)())
            0: ──H──╭SWAP──╭SWAP──Y──┤ ⟨Z⟩
            1: ──X──╰SWAP──│─────────┤
            2: ────────────╰SWAP─────┤


        We can remove the SWAP gates by running the ``undo_swap`` transform:

        >>> optimized_qfunc = undo_swaps(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)())
            0: ──Y──┤ ⟨Z⟩
            1: ──H──┤
            2: ──X──┤

    