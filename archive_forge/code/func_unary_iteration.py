import abc
from typing import Callable, Dict, Iterator, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_function
@deprecated_cirq_ft_function()
def unary_iteration(l_iter: int, r_iter: int, flanking_ops: List[cirq.Operation], controls: Sequence[cirq.Qid], selection: Sequence[cirq.Qid], qubit_manager: cirq.QubitManager, break_early: Callable[[int, int], bool]=lambda l, r: False) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    """The method performs unary iteration on `selection` integer in `range(l_iter, r_iter)`.

    Unary iteration is a coherent for loop that can be used to conditionally perform a different
    operation on a target register for every integer in the `range(l_iter, r_iter)` stored in the
    selection register.

    Users can write multi-dimensional coherent for loops as follows:

    >>> import cirq
    >>> from cirq_ft import unary_iteration
    >>> N, M = 5, 7
    >>> target = [[cirq.q(f't({i}, {j})') for j in range(M)] for i in range(N)]
    >>> selection = [[cirq.q(f's({i}, {j})') for j in range(3)] for i in range(3)]
    >>> circuit = cirq.Circuit()
    >>> i_ops = []
    >>> qm = cirq.GreedyQubitManager("ancilla", maximize_reuse=True)
    >>> for i_optree, i_ctrl, i in unary_iteration(0, N, i_ops, [], selection[0], qm):
    ...     circuit.append(i_optree)
    ...     j_ops = []
    ...     for j_optree, j_ctrl, j in unary_iteration(0, M, j_ops, [i_ctrl], selection[1], qm):
    ...         circuit.append(j_optree)
    ...         # Conditionally perform operations on target register using `j_ctrl`, `i` & `j`.
    ...         circuit.append(cirq.CNOT(j_ctrl, target[i][j]))
    ...     circuit.append(j_ops)
    >>> circuit.append(i_ops)

    Note: Unary iteration circuits assume that the selection register stores integers only in the
    range `[l, r)` for which the corresponding unary iteration circuit should be built.

    Args:
        l_iter: Starting index of the iteration range.
        r_iter: Ending index of the iteration range.
        flanking_ops: A list of `cirq.Operation`s that represents operations to be inserted in the
            circuit before/after the first/last iteration of the unary iteration for loop. Note that
            the list is mutated by the function, such that before calling the function, the list
            represents operations to be inserted before the first iteration and after the last call
            to the function, list represents operations to be inserted at the end of last iteration.
        controls: Control register of qubits.
        selection: Selection register of qubits.
        qubit_manager: A `cirq.QubitManager` to allocate new qubits.
        break_early: For each internal node of the segment tree, `break_early(l, r)` is called to
            evaluate whether the unary iteration should terminate early and not recurse in the
            subtree of the node representing range `[l, r)`. If True, the internal node is
            considered equivalent to a leaf node and the method yields only one tuple
            `(OP_TREE, control_qubit, l)` for all integers in the range `[l, r)`.

    Yields:
        (r_iter - l_iter) different tuples, each corresponding to an integer in range
        [l_iter, r_iter).
        Each returned tuple also corresponds to a unique leaf in the unary iteration tree.
        The values of yielded `Tuple[cirq.OP_TREE, cirq.Qid, int]` correspond to:
        - cirq.OP_TREE: The op-tree to be inserted in the circuit to get to the current leaf.
        - cirq.Qid: Control qubit used to conditionally apply operations on the target conditioned
            on the returned integer.
        - int: The current integer in the iteration `range(l_iter, r_iter)`.
    """
    assert 2 ** len(selection) >= r_iter - l_iter
    assert len(selection) > 0
    ancilla = qubit_manager.qalloc(max(0, len(controls) + len(selection) - 1))
    if len(controls) == 0:
        yield from _unary_iteration_zero_control(flanking_ops, selection, ancilla, l_iter, r_iter, break_early)
    elif len(controls) == 1:
        yield from _unary_iteration_single_control(flanking_ops, controls[0], selection, ancilla, l_iter, r_iter, break_early)
    else:
        yield from _unary_iteration_multi_controls(flanking_ops, controls, selection, ancilla, l_iter, r_iter, break_early)
    qubit_manager.qfree(ancilla)