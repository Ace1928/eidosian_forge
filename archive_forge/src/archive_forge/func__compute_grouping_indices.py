import itertools
import numbers
from collections.abc import Iterable
from copy import copy
import functools
from typing import List
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import Observable, Tensor
from pennylane.wires import Wires
def _compute_grouping_indices(observables, grouping_type='qwc', method='rlf'):
    observable_groups = qml.pauli.group_observables(observables, coefficients=None, grouping_type=grouping_type, method=method)
    observables = copy(observables)
    indices = []
    available_indices = list(range(len(observables)))
    for partition in observable_groups:
        indices_this_group = []
        for pauli_word in partition:
            for ind, observable in enumerate(observables):
                if qml.pauli.are_identical_pauli_words(pauli_word, observable):
                    indices_this_group.append(available_indices[ind])
                    observables.pop(ind)
                    available_indices.pop(ind)
                    break
        indices.append(tuple(indices_this_group))
    return tuple(indices)