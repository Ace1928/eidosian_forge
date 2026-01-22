import dataclasses
from typing import Any, Dict, List, Sequence, Set, Type, TypeVar, Union
import numpy as np
import cirq, cirq_google
from cirq import _compat, devices
from cirq.devices import noise_utils
from cirq.transformers.heuristic_decompositions import gate_tabulation_math_utils
@_compat.cached_property
def _depolarizing_error(self) -> Dict[noise_utils.OpIdentifier, float]:
    depol_errors = super()._depolarizing_error

    def extract_entangling_error(match_id: noise_utils.OpIdentifier):
        """Gets the entangling error component of depol_errors[match_id]."""
        unitary_err = cirq.unitary(self.fsim_errors[match_id])
        fid = gate_tabulation_math_utils.unitary_entanglement_fidelity(unitary_err, np.eye(4))
        return 1 - fid
    for op_id in depol_errors:
        if op_id.gate_type not in self.two_qubit_gates():
            continue
        if op_id in self.fsim_errors:
            depol_errors[op_id] -= extract_entangling_error(op_id)
            continue
        match_id = None
        candidate_parents = [parent_id for parent_id in self.fsim_errors if op_id.is_proper_subtype_of(parent_id)]
        for parent_id in candidate_parents:
            if match_id is None or parent_id.is_proper_subtype_of(match_id):
                match_id = parent_id
        if match_id is not None:
            depol_errors[op_id] -= extract_entangling_error(match_id)
    return depol_errors