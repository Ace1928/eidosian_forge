from functools import partial
from typing import Sequence, Callable
from collections import OrderedDict
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.op_transforms import OperationTransformError
from pennylane.transforms import transform
from pennylane.wires import Wires
def to_zx(tape, expand_measurements=False):
    """This transform converts a PennyLane quantum tape to a ZX-Graph in the `PyZX framework <https://pyzx.readthedocs.io/en/latest/>`_.
    The graph can be optimized and transformed by well-known ZX-calculus reductions.

    Args:
        tape(QNode or QuantumTape or Callable or Operation): The PennyLane quantum circuit.
        expand_measurements(bool): The expansion will be applied on measurements that are not in the Z-basis and
            rotations will be added to the operations.

    Returns:
        graph (pyzx.Graph) or qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the ZX graph in the form of a PyZX graph.

    **Example**

    You can use the transform decorator directly on your :class:`~.QNode`, quantum function and executing it will produce a
    PyZX graph. You can also use the transform directly on the :class:`~.QuantumTape`.

    .. code-block:: python

        import pyzx
        dev = qml.device('default.qubit', wires=2)

        @qml.transforms.to_zx
        @qml.qnode(device=dev)
        def circuit(p):
            qml.RZ(p[0], wires=1),
            qml.RZ(p[1], wires=1),
            qml.RX(p[2], wires=0),
            qml.Z(0),
            qml.RZ(p[3], wires=1),
            qml.X(1),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 0]),
            qml.SWAP(wires=[0, 1]),
            return qml.expval(qml.Z(0) @ qml.Z(1))

        params = [5 / 4 * np.pi, 3 / 4 * np.pi, 0.1, 0.3]
        g = circuit(params)

    >>> g
    Graph(20 vertices, 23 edges)

    It is now a PyZX graph and can apply function from the framework on your Graph, for example you can draw it:

    >>> pyzx.draw_matplotlib(g)
    <Figure size 800x200 with 1 Axes>

    Alternatively you can use the transform directly on a quantum tape and get PyZX graph.

    .. code-block:: python

        operations = [
                qml.RZ(5 / 4 * np.pi, wires=1),
                qml.RZ(3 / 4 * np.pi, wires=1),
                qml.RX(0.1, wires=0),
                qml.Z(0),
                qml.RZ(0.3, wires=1),
                qml.X(1),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 0]),
                qml.SWAP(wires=[0, 1]),
            ]

        tape = qml.tape.QuantumTape(operations)
        g = qml.transforms.to_zx(tape)

    >>> g
    Graph(20 vertices, 23 edges)

    .. details::
        :title: Usage Details

        Here we give an example of how to use optimization techniques from ZX calculus to reduce the T count of a
        quantum circuit and get back a PennyLane circuit.

        Let's start by starting with the mod 5 4 circuit from a known benchmark `library <https://github.com/njross/optimizer>`_
        the expanded circuit before optimization is the following QNode:

        .. code-block:: python

            dev = qml.device("default.qubit", wires=5)

            @qml.transforms.to_zx
            @qml.qnode(device=dev)
            def mod_5_4():
                qml.X(4),
                qml.Hadamard(wires=4),
                qml.CNOT(wires=[3, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[0, 4]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[3, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[0, 4]),
                qml.T(wires=[3]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[0, 3]),
                qml.T(wires=[0]),
                qml.adjoint(qml.T(wires=[3]))
                qml.CNOT(wires=[0, 3]),
                qml.CNOT(wires=[3, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[2, 4]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[3, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[2, 4]),
                qml.T(wires=[3]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[2, 3]),
                qml.T(wires=[2]),
                qml.adjoint(qml.T(wires=[3]))
                qml.CNOT(wires=[2, 3]),
                qml.Hadamard(wires=[4]),
                qml.CNOT(wires=[3, 4]),
                qml.Hadamard(wires=4),
                qml.CNOT(wires=[2, 4]),
                qml.adjoint(qml.T(wires=[4]),)
                qml.CNOT(wires=[1, 4]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[2, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[1, 4]),
                qml.T(wires=[4]),
                qml.T(wires=[2]),
                qml.CNOT(wires=[1, 2]),
                qml.T(wires=[1]),
                qml.adjoint(qml.T(wires=[2]))
                qml.CNOT(wires=[1, 2]),
                qml.Hadamard(wires=[4]),
                qml.CNOT(wires=[2, 4]),
                qml.Hadamard(wires=4),
                qml.CNOT(wires=[1, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[0, 4]),
                qml.T(wires=[4]),
                qml.CNOT(wires=[1, 4]),
                qml.adjoint(qml.T(wires=[4])),
                qml.CNOT(wires=[0, 4]),
                qml.T(wires=[4]),
                qml.T(wires=[1]),
                qml.CNOT(wires=[0, 1]),
                qml.T(wires=[0]),
                qml.adjoint(qml.T(wires=[1])),
                qml.CNOT(wires=[0, 1]),
                qml.Hadamard(wires=[4]),
                qml.CNOT(wires=[1, 4]),
                qml.CNOT(wires=[0, 4]),
                return qml.expval(qml.Z(0))

        The circuit contains 63 gates; 28 :func:`qml.T` gates, 28 :func:`qml.CNOT`, 6 :func:`qml.Hadmard` and
        1 :func:`qml.X`. We applied the ``qml.transforms.to_zx`` decorator in order to transform our circuit to
        a ZX graph.

        You can get the PyZX graph by simply calling the QNode:

        >>> g = mod_5_4()
        >>> pyzx.tcount(g)
        28

        PyZX gives multiple options for optimizing ZX graphs (:func:`pyzx.full_reduce`, :func:`pyzx.teleport_reduce`, ...).
        The :func:`pyzx.full_reduce` applies all optimization passes, but the final result may not be circuit-like.
        Converting back to a quantum circuit from a fully reduced graph may be difficult to impossible.
        Therefore we instead recommend using :func:`pyzx.teleport_reduce`, as it preserves the circuit structure.

        >>> g = pyzx.simplify.teleport_reduce(g)
        >>> pyzx.tcount(g)
        8

        If you give a closer look, the circuit contains now 53 gates; 8 :func:`qml.T` gates, 28 :func:`qml.CNOT`, 6 :func:`qml.Hadmard` and
        1 :func:`qml.X` and 10 :func:`qml.S`. We successfully reduced the T-count by 20 and have ten additional
        S gates. The number of CNOT gates remained the same.

        The :func:`from_zx` transform can now convert the optimized circuit back into PennyLane operations:

        .. code-block:: python

            tape_opt = qml.transforms.from_zx(g)

            wires = qml.wires.Wires([4, 3, 0, 2, 1])
            wires_map = dict(zip(tape_opt.wires, wires))
            tapes_opt_reorder, fn = qml.map_wires(input=tape_opt, wire_map=wires_map)[0][0]
            tape_opt_reorder = fn(tapes_opt_reorder)

            @qml.qnode(device=dev)
            def mod_5_4():
                for g in tape_opt_reorder:
                    qml.apply(g)
                return qml.expval(qml.Z(0))

        >>> mod_5_4()
        tensor(1., requires_grad=True)

    .. note::

        It is a PennyLane adapted and reworked `circuit_to_graph <https://github.com/Quantomatic/pyzx/blob/master/pyzx/circuit/graphparser.py>`_
        function.

        Copyright (C) 2018 - Aleks Kissinger and John van de Wetering
    """
    if not isinstance(tape, Operator):
        if not isinstance(tape, (qml.tape.QuantumScript, qml.QNode)) and (not callable(tape)):
            raise OperationTransformError('Input is not an Operator, tape, QNode, or quantum function')
        return _to_zx_transform(tape, expand_measurements=expand_measurements)
    return to_zx(QuantumScript([tape]))