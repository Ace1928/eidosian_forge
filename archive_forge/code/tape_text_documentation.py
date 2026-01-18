from dataclasses import dataclass
from typing import Optional
import pennylane as qml
from pennylane.measurements import (
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
Text based diagram for a Quantum Tape.

    Args:
        tape (QuantumTape): the operations and measurements to draw

    Keyword Args:
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        decimals (int): How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.
        max_length (Int) : Maximum length of a individual line.  After this length, the diagram will
            begin anew beneath the previous lines.
        show_matrices=True (bool): show matrix valued parameters below all circuit diagrams
        cache (dict): Used to store information between recursive calls. Necessary keys are ``'tape_offset'``
            and ``'matrices'``.

    Returns:
        str : String based graphic of the circuit.

    **Example:**

    .. code-block:: python

        ops = [
            qml.QFT(wires=(0, 1, 2)),
            qml.RX(1.234, wires=0),
            qml.RY(1.234, wires=1),
            qml.RZ(1.234, wires=2),
            qml.Toffoli(wires=(0, 1, "aux"))
        ]
        measurements = [
            qml.expval(qml.Z("aux")),
            qml.var(qml.Z(0) @ qml.Z(1)),
            qml.probs(wires=(0, 1, 2, "aux"))
        ]
        tape = qml.tape.QuantumTape(ops, measurements)

    >>> print(qml.drawer.tape_text(tape))
      0: ─╭QFT──RX─╭●─┤ ╭Var[Z@Z] ╭Probs
      1: ─├QFT──RY─├●─┤ ╰Var[Z@Z] ├Probs
      2: ─╰QFT──RZ─│──┤           ├Probs
    aux: ──────────╰X─┤  <Z>      ╰Probs

    .. details::
        :title: Usage Details

    By default, parameters are omitted. By specifying the ``decimals`` keyword, parameters
    are displayed to the specified precision. Matrix-valued parameters are never displayed.

    >>> print(qml.drawer.tape_text(tape, decimals=2))
      0: ─╭QFT──RX(1.23)─╭●─┤ ╭Var[Z@Z] ╭Probs
      1: ─├QFT──RY(1.23)─├●─┤ ╰Var[Z@Z] ├Probs
      2: ─╰QFT──RZ(1.23)─│──┤           ├Probs
    aux: ────────────────╰X─┤  <Z>      ╰Probs


    The ``max_length`` keyword wraps long circuits:

    .. code-block:: python

        rng = np.random.default_rng(seed=42)
        shape = qml.StronglyEntanglingLayers.shape(n_wires=5, n_layers=5)
        params = rng.random(shape)
        tape2 = qml.StronglyEntanglingLayers(params, wires=range(5)).expand()
        print(qml.drawer.tape_text(tape2, max_length=60))


    .. code-block:: none

        0: ──Rot─╭●──────────╭X──Rot─╭●───────╭X──Rot──────╭●────╭X
        1: ──Rot─╰X─╭●───────│───Rot─│──╭●────│──╭X────Rot─│──╭●─│─
        2: ──Rot────╰X─╭●────│───Rot─╰X─│──╭●─│──│─────Rot─│──│──╰●
        3: ──Rot───────╰X─╭●─│───Rot────╰X─│──╰●─│─────Rot─╰X─│────
        4: ──Rot──────────╰X─╰●──Rot───────╰X────╰●────Rot────╰X───

        ───Rot───────────╭●─╭X──Rot──────╭●──────────────╭X─┤
        ──╭X────Rot──────│──╰●─╭X────Rot─╰X───╭●─────────│──┤
        ──│────╭X────Rot─│─────╰●───╭X────Rot─╰X───╭●────│──┤
        ──╰●───│─────Rot─│──────────╰●───╭X────Rot─╰X─╭●─│──┤
        ───────╰●────Rot─╰X──────────────╰●────Rot────╰X─╰●─┤


    The ``wire_order`` keyword specifies the order of the wires from
    top to bottom:

    >>> print(qml.drawer.tape_text(tape, wire_order=["aux", 2, 1, 0]))
    aux: ──────────╭X─┤  <Z>      ╭Probs
      2: ─╭QFT──RZ─│──┤           ├Probs
      1: ─├QFT──RY─├●─┤ ╭Var[Z@Z] ├Probs
      0: ─╰QFT──RX─╰●─┤ ╰Var[Z@Z] ╰Probs

    If the wire order contains empty wires, they are only shown if the ``show_all_wires=True``.

    >>> print(qml.drawer.tape_text(tape, wire_order=["a", "b", "aux", 0, 1, 2], show_all_wires=True))
      a: ─────────────┤
      b: ─────────────┤
    aux: ──────────╭X─┤  <Z>      ╭Probs
      0: ─╭QFT──RX─├●─┤ ╭Var[Z@Z] ├Probs
      1: ─├QFT──RY─╰●─┤ ╰Var[Z@Z] ├Probs
      2: ─╰QFT──RZ────┤           ╰Probs

    Matrix valued parameters are always denoted by ``M`` followed by an integer corresponding to
    unique matrices.  The list of unique matrices can be printed at the end of the diagram by
    selecting ``show_matrices=True`` (the default):

    .. code-block:: python

        ops = [
            qml.QubitUnitary(np.eye(2), wires=0),
            qml.QubitUnitary(np.eye(2), wires=1)
        ]
        measurements = [qml.expval(qml.Hermitian(np.eye(4), wires=(0,1)))]
        tape = qml.tape.QuantumTape(ops, measurements)

    >>> print(qml.drawer.tape_text(tape))
    0: ──U(M0)─┤ ╭<𝓗(M1)>
    1: ──U(M0)─┤ ╰<𝓗(M1)>
    M0 =
    [[1. 0.]
    [0. 1.]]
    M1 =
    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]]

    An existing matrix cache can be passed via the ``cache`` keyword. Note that the dictionary
    passed to ``cache`` will be modified during execution to contain any new matrices and the
    tape offset.

    >>> cache = {'matrices': [-np.eye(3)]}
    >>> print(qml.drawer.tape_text(tape, cache=cache))
    0: ──U(M1)─┤ ╭<𝓗(M2)>
    1: ──U(M1)─┤ ╰<𝓗(M2)>
    M0 =
    [[-1. -0. -0.]
    [-0. -1. -0.]
    [-0. -0. -1.]]
    M1 =
    [[1. 0.]
    [0. 1.]]
    M2 =
    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]]
    >>> cache
    {'matrices': [tensor([[-1., -0., -0.],
        [-0., -1., -0.],
        [-0., -0., -1.]], requires_grad=True), tensor([[1., 0.],
        [0., 1.]], requires_grad=True), tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]], requires_grad=True)], 'tape_offset': 0}

    When the provided tape has nested tapes inside, this function is called recursively.
    To maintain numbering of tapes to arbitrary levels of nesting, the ``cache`` keyword
    uses the ``"tape_offset"`` value to determine numbering. Note that the value is updated
    during the call.

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            with qml.tape.QuantumTape() as tape_inner:
                qml.X(0)

        cache = {'tape_offset': 3}
        print(qml.drawer.tape_text(tape, cache=cache))
        print("New tape offset: ", cache['tape_offset'])


    .. code-block:: none

        0: ──Tape:3─┤

        Tape:3
        0: ──X─┤
        New tape offset:  4

    