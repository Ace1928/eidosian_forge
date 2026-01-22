from typing import Tuple as tTuple
from collections import defaultdict
from functools import cmp_to_key, reduce
from itertools import product
import operator
from .sympify import sympify
from .basic import Basic
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .traversal import bottom_up
from sympy.utilities.iterables import sift
from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add
@staticmethod
def _matches_noncomm(nodes, targets, repl_dict=None):
    """Non-commutative multiplication matcher.

        `nodes` is a list of symbols within the matcher multiplication
        expression, while `targets` is a list of arguments in the
        multiplication expression being matched against.
        """
    if repl_dict is None:
        repl_dict = {}
    else:
        repl_dict = repl_dict.copy()
    agenda = []
    state = (0, 0)
    node_ind, target_ind = state
    wildcard_dict = {}
    while target_ind < len(targets) and node_ind < len(nodes):
        node = nodes[node_ind]
        if node.is_Wild:
            Mul._matches_add_wildcard(wildcard_dict, state)
        states_matches = Mul._matches_new_states(wildcard_dict, state, nodes, targets)
        if states_matches:
            new_states, new_matches = states_matches
            agenda.extend(new_states)
            if new_matches:
                for match in new_matches:
                    repl_dict[match] = new_matches[match]
        if not agenda:
            return None
        else:
            state = agenda.pop()
            node_ind, target_ind = state
    return repl_dict