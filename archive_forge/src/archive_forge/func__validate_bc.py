import numpy as np
from . import PPoly
from ._polyint import _isscalar
from scipy.linalg import solve_banded, solve
@staticmethod
def _validate_bc(bc_type, y, expected_deriv_shape, axis):
    """Validate and prepare boundary conditions.

        Returns
        -------
        validated_bc : 2-tuple
            Boundary conditions for a curve start and end.
        y : ndarray
            y casted to complex dtype if one of the boundary conditions has
            complex dtype.
        """
    if isinstance(bc_type, str):
        if bc_type == 'periodic':
            if not np.allclose(y[0], y[-1], rtol=1e-15, atol=1e-15):
                raise ValueError(f"The first and last `y` point along axis {axis} must be identical (within machine precision) when bc_type='periodic'.")
        bc_type = (bc_type, bc_type)
    else:
        if len(bc_type) != 2:
            raise ValueError('`bc_type` must contain 2 elements to specify start and end conditions.')
        if 'periodic' in bc_type:
            raise ValueError("'periodic' `bc_type` is defined for both curve ends and cannot be used with other boundary conditions.")
    validated_bc = []
    for bc in bc_type:
        if isinstance(bc, str):
            if bc == 'clamped':
                validated_bc.append((1, np.zeros(expected_deriv_shape)))
            elif bc == 'natural':
                validated_bc.append((2, np.zeros(expected_deriv_shape)))
            elif bc in ['not-a-knot', 'periodic']:
                validated_bc.append(bc)
            else:
                raise ValueError(f'bc_type={bc} is not allowed.')
        else:
            try:
                deriv_order, deriv_value = bc
            except Exception as e:
                raise ValueError('A specified derivative value must be given in the form (order, value).') from e
            if deriv_order not in [1, 2]:
                raise ValueError('The specified derivative order must be 1 or 2.')
            deriv_value = np.asarray(deriv_value)
            if deriv_value.shape != expected_deriv_shape:
                raise ValueError('`deriv_value` shape {} is not the expected one {}.'.format(deriv_value.shape, expected_deriv_shape))
            if np.issubdtype(deriv_value.dtype, np.complexfloating):
                y = y.astype(complex, copy=False)
            validated_bc.append((deriv_order, deriv_value))
    return (validated_bc, y)