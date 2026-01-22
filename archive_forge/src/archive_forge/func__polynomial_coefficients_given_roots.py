from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def _polynomial_coefficients_given_roots(roots):
    """
    Given the `roots` of a polynomial, find the polynomial's coefficients.

    If roots = (r_1, ..., r_n), then the method returns
    coefficients (a_0, a_1, ..., a_n (== 1)) so that
    p(x) = (x - r_1) * ... * (x - r_n)
         = x^n + a_{n-1} * x^{n-1} + ... a_1 * x_1 + a_0

    Note: for better performance requires writing a low-level kernel
    """
    poly_order = roots.shape[-1]
    poly_coeffs_shape = list(roots.shape)
    poly_coeffs_shape[-1] += 2
    poly_coeffs = roots.new_zeros(poly_coeffs_shape)
    poly_coeffs[..., 0] = 1
    poly_coeffs[..., -1] = 1
    for i in range(1, poly_order + 1):
        poly_coeffs_new = poly_coeffs.clone() if roots.requires_grad else poly_coeffs
        out = poly_coeffs_new.narrow(-1, poly_order - i, i + 1)
        out -= roots.narrow(-1, i - 1, 1) * poly_coeffs.narrow(-1, poly_order - i + 1, i + 1)
        poly_coeffs = poly_coeffs_new
    return poly_coeffs.narrow(-1, 1, poly_order + 1)