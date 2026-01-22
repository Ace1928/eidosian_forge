import functools
import itertools
from string import ascii_letters as ABC
from autoray import numpy as np
from numpy import float64
import pennylane as qml
from . import single_dispatch  # pylint:disable=unused-import
from .matrix_manipulation import _permute_dense_matrix
from .multi_dispatch import diag, dot, scatter_element_add, einsum, get_interface
from .utils import is_abstract, allclose, cast, convert_like, cast_like
def _batched_partial_trace_nonrep_indices(matrix, is_batched, indices, batch_dim, dim):
    """Compute the reduced density matrix for autograd interface by tracing out the provided indices with the use
    of projectors as same subscripts indices are not supported in autograd backprop.
    """
    num_indices = int(np.log2(dim))
    rho_dim = 2 * num_indices
    matrix = np.reshape(matrix, [batch_dim] + [2] * 2 * num_indices)
    kraus = cast(np.eye(2), matrix.dtype)
    kraus = np.reshape(kraus, (2, 1, 2))
    kraus_dagger = np.asarray([np.conj(np.transpose(k)) for k in kraus])
    kraus = convert_like(kraus, matrix)
    kraus_dagger = convert_like(kraus_dagger, matrix)
    for target_wire in indices:
        state_indices = ABC[1:rho_dim + 1]
        row_wires_list = [target_wire + 1]
        row_indices = ''.join(ABC_ARRAY[row_wires_list].tolist())
        col_wires_list = [w + num_indices for w in row_wires_list]
        col_indices = ''.join(ABC_ARRAY[col_wires_list].tolist())
        num_partial_trace_wires = 1
        new_row_indices = ABC[rho_dim + 1:rho_dim + num_partial_trace_wires + 1]
        new_col_indices = ABC[rho_dim + num_partial_trace_wires + 1:rho_dim + 2 * num_partial_trace_wires + 1]
        kraus_index = ABC[rho_dim + 2 * num_partial_trace_wires + 1:rho_dim + 2 * num_partial_trace_wires + 2]
        new_state_indices = functools.reduce(lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]), zip(col_indices + row_indices, new_col_indices + new_row_indices), state_indices)
        einsum_indices = f'{kraus_index}{new_row_indices}{row_indices}, a{state_indices},{kraus_index}{col_indices}{new_col_indices}->a{new_state_indices}'
        matrix = einsum(einsum_indices, kraus, matrix, kraus_dagger)
    number_wires_sub = num_indices - len(indices)
    reduced_density_matrix = np.reshape(matrix, (batch_dim, 2 ** number_wires_sub, 2 ** number_wires_sub))
    return reduced_density_matrix if is_batched else reduced_density_matrix[0]