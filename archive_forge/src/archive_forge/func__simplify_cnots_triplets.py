import functools
import itertools
from typing import Any, Dict, Generator, List, Sequence, Tuple
import sympy.parsing.sympy_parser as sympy_parser
import cirq
from cirq import value
from cirq.ops import raw_types
from cirq.ops.linear_combinations import PauliSum, PauliString
def _simplify_cnots_triplets(cnots: List[Tuple[int, int]], flip_control_and_target: bool) -> Tuple[bool, List[Tuple[int, int]]]:
    """Simplifies CNOT pairs according to equation 11 of [4].

    CNOT(i, j) @ CNOT(j, k) == CNOT(j, k) @ CNOT(i, k) @ CNOT(i, j)
    ───@───────       ───────@───@───
       │                     │   │
    ───X───@───   =   ───@───┼───X───
           │             │   │
    ───────X───       ───X───X───────

    Args:
        cnots: A list of CNOTS, encoded as integer tuples (control, target).
        flip_control_and_target: Whether to flip control and target.

    Returns:
        A tuple containing a Boolean that tells whether a simplification has been performed and the
        CNOT list, potentially simplified, encoded as integer tuples (control, target).
    """
    target, control = (0, 1) if flip_control_and_target else (1, 0)
    for j in range(1, len(cnots) - 1):
        prev_match_index: Dict[int, int] = {}
        for i in range(j - 1, -1, -1):
            if cnots[i][target] == cnots[j][target]:
                continue
            if cnots[i][control] != cnots[j][control]:
                break
            prev_match_index[cnots[i][target]] = i
        post_match_index: Dict[int, int] = {}
        for k in range(j + 1, len(cnots)):
            if cnots[j][control] == cnots[k][control]:
                continue
            if cnots[j][target] != cnots[k][target]:
                break
            post_match_index[cnots[k][control]] = k
        keys = prev_match_index.keys() & post_match_index.keys()
        for key in keys:
            new_idx: List[int] = [idx for idx in range(j) if idx != prev_match_index[key]] + [post_match_index[key], prev_match_index[key]] + [idx for idx in range(j + 1, len(cnots)) if idx != post_match_index[key]]
            cnots = [cnots[idx] for idx in new_idx]
            return (True, cnots)
    return (False, cnots)