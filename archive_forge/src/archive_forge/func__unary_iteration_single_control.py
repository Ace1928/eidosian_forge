import abc
from typing import Callable, Dict, Iterator, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_function
def _unary_iteration_single_control(ops: List[cirq.Operation], control: cirq.Qid, selection: Sequence[cirq.Qid], ancilla: Sequence[cirq.Qid], l_iter: int, r_iter: int, break_early: Callable[[int, int], bool]) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    sl, l, r = (0, 0, 2 ** len(selection))
    yield from _unary_iteration_segtree(ops, control, selection, ancilla, sl, l, r, l_iter, r_iter, break_early)