from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from .parametrized_hamiltonian import ParametrizedHamiltonian
def _reorder_parameters(params, coeffs_parametrized):
    """Takes `params`, and reorganizes it based on whether the Hamiltonian has
    callable phase and/or callable amplitude.
    Consolidates phase and amplitude parameters in the case that both are callable,
    and duplicates phase and/or amplitude parameters if either are callables, since
    they will be passed to two operators in the Hamiltonian"""
    reordered_params = []
    coeff_idx = 0
    params_idx = 0
    for i, coeff in enumerate(coeffs_parametrized):
        if i == coeff_idx:
            if isinstance(coeff, AmplitudeAndPhase):
                if coeff.phase_is_callable and coeff.amp_is_callable:
                    reordered_params.append([params[params_idx], params[params_idx + 1]])
                    reordered_params.append([params[params_idx], params[params_idx + 1]])
                    coeff_idx += 2
                    params_idx += 2
                elif coeff.phase_is_callable or coeff.amp_is_callable:
                    reordered_params.append(params[params_idx])
                    reordered_params.append(params[params_idx])
                    coeff_idx += 2
                    params_idx += 1
            else:
                reordered_params.append(params[params_idx])
                coeff_idx += 1
                params_idx += 1
    return reordered_params