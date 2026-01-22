import functools
import itertools
from typing import Any, Dict, Generator, List, Sequence, Tuple
import sympy.parsing.sympy_parser as sympy_parser
import cirq
from cirq import value
from cirq.ops import raw_types
from cirq.ops.linear_combinations import PauliSum, PauliString
def _simplify_cnots(cnots: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Takes a series of CNOTs and tries to applies rule to cancel out gates.

    Algorithm based on "Efficient quantum circuits for diagonal unitaries without ancillas" by
    Jonathan Welch, Daniel Greenbaum, Sarah Mostame, Alán Aspuru-Guzik
    https://arxiv.org/abs/1306.3991

    Args:
        cnots: A list of CNOTs represented as tuples of integer (control, target).

    Returns:
        The simplified list of CNOTs, encoded as integer tuples (control, target).
    """
    found_simplification = True
    while found_simplification:
        for simplify_fn, flip_control_and_target in itertools.product([_simplify_commuting_cnots, _simplify_cnots_triplets], [False, True]):
            found_simplification, cnots = simplify_fn(cnots, flip_control_and_target)
            if found_simplification:
                break
    return cnots