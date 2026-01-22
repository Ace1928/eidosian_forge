import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
def _add_match(match_list, backward_match_list):
    """
    Add a match configuration found by pattern matching if it is not already in final list of matches.
    If the match is already in the final list, the qubit configuration is added to the existing Match.
    Args:
        match_list (list): match from the backward part of the algorithm.
        backward_match_list (list): List of matches found by the algorithm for a given configuration.
    """
    already_in = False
    for b_match in backward_match_list:
        for l_match in match_list:
            if b_match.match == l_match.match:
                index = match_list.index(l_match)
                match_list[index].qubit.append(b_match.qubit[0])
                already_in = True
        if not already_in:
            match_list.append(b_match)