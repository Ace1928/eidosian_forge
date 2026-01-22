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
def _matches_new_states(dictionary, state, nodes, targets):
    node_ind, target_ind = state
    node = nodes[node_ind]
    target = targets[target_ind]
    if target_ind >= len(targets) - 1 and node_ind < len(nodes) - 1:
        return None
    if node.is_Wild:
        match_attempt = Mul._matches_match_wilds(dictionary, node_ind, nodes, targets)
        if match_attempt:
            other_node_inds = Mul._matches_get_other_nodes(dictionary, nodes, node_ind)
            for ind in other_node_inds:
                other_begin, other_end = dictionary[ind]
                curr_begin, curr_end = dictionary[node_ind]
                other_targets = targets[other_begin:other_end + 1]
                current_targets = targets[curr_begin:curr_end + 1]
                for curr, other in zip(current_targets, other_targets):
                    if curr != other:
                        return None
            new_state = [(node_ind, target_ind + 1)]
            if node_ind < len(nodes) - 1:
                new_state.append((node_ind + 1, target_ind + 1))
            return (new_state, match_attempt)
    else:
        if node_ind >= len(nodes) - 1 and target_ind < len(targets) - 1:
            return None
        match_attempt = node.matches(target)
        if match_attempt:
            return ([(node_ind + 1, target_ind + 1)], match_attempt)
        elif node == target:
            return ([(node_ind + 1, target_ind + 1)], None)
        else:
            return None