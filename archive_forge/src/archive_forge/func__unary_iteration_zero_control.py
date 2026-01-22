import abc
from typing import Callable, Dict, Iterator, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_function
def _unary_iteration_zero_control(ops: List[cirq.Operation], selection: Sequence[cirq.Qid], ancilla: Sequence[cirq.Qid], l_iter: int, r_iter: int, break_early: Callable[[int, int], bool]) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    sl, l, r = (0, 0, 2 ** len(selection))
    m = l + r >> 1
    ops.append(cirq.X(selection[0]))
    yield from _unary_iteration_segtree(ops, selection[0], selection[1:], ancilla, sl, l, m, l_iter, r_iter, break_early)
    ops.append(cirq.X(selection[0]))
    yield from _unary_iteration_segtree(ops, selection[0], selection[1:], ancilla, sl, m, r, l_iter, r_iter, break_early)