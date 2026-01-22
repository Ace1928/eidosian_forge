from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.measurements import VarianceMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .finite_difference import finite_diff
from .general_shift_rules import (
from .gradient_transform import (
def _single_variance_gradient(tape, var_mask, pdA2, f0, pdA):
    """Auxiliary function to return the derivative of variances.

    return d(var(A))/dp = d<A^2>/dp -2 * <A> * d<A>/dp for the variances (var_mask==True)
    d<A>/dp for plain expectations (var_mask==False)

    Note: if isinstance(pdA2, int) and pdA2 != 0, then len(pdA2) == len(pdA)

    """
    num_params = len(tape.trainable_params)
    num_measurements = len(tape.measurements)
    if num_measurements > 1:
        if num_params == 1:
            var_grad = []
            for m_idx in range(num_measurements):
                m_res = pdA[m_idx]
                if var_mask[m_idx]:
                    pdA2_comp = pdA2[m_idx] if pdA2 != 0 else pdA2
                    f0_comp = f0[m_idx]
                    m_res = _get_var_with_second_order(pdA2_comp, f0_comp, m_res)
                var_grad.append(m_res)
            return tuple(var_grad)
        var_grad = []
        for m_idx in range(num_measurements):
            m_res = []
            if var_mask[m_idx]:
                for p_idx in range(num_params):
                    pdA2_comp = pdA2[m_idx][p_idx] if pdA2 != 0 else pdA2
                    f0_comp = f0[m_idx]
                    pdA_comp = pdA[m_idx][p_idx]
                    r = _get_var_with_second_order(pdA2_comp, f0_comp, pdA_comp)
                    m_res.append(r)
                m_res = tuple(m_res)
            else:
                m_res = tuple((pdA[m_idx][p_idx] for p_idx in range(num_params)))
            var_grad.append(m_res)
        return tuple(var_grad)
    if num_params == 1:
        return _get_var_with_second_order(pdA2, f0, pdA)
    var_grad = []
    for p_idx in range(num_params):
        pdA2_comp = pdA2[p_idx] if pdA2 != 0 else pdA2
        r = _get_var_with_second_order(pdA2_comp, f0, pdA[p_idx])
        var_grad.append(r)
    return tuple(var_grad)