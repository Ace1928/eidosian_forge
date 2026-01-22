import functools
import itertools
from operator import mul
from typing import Dict, List, Iterable, Sequence, Set, Tuple, Union, cast
import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
from pyquil.experiment._main import Experiment
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, TensorProductState, _OneQState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.paulis import PauliTerm, sI
from pyquil.quil import Program
def _max_weight_state(states: Iterable[TensorProductState]) -> Union[None, TensorProductState]:
    """Construct a TensorProductState by taking the single-qubit state at each
    qubit position.

    This function will return ``None`` if the input states are not compatible

    For example, the max_weight_state of ["(+X, q0)", "(-Z, q1)"] is "(+X, q0; -Z q1)". Asking for
    the max weight state of something like ["(+X, q0)", "(+Z, q0)"] will return None.
    """
    mapping = dict()
    for state in states:
        for oneq_state in state.states:
            if oneq_state.qubit in mapping:
                if mapping[oneq_state.qubit] != oneq_state:
                    return None
            else:
                mapping[oneq_state.qubit] = oneq_state
    return TensorProductState(list(mapping.values()))