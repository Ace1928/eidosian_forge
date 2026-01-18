import numpy as np
from cvxpy.lin_ops.tree_mat import mul, sum_dicts, tmul
def vec_to_dict(vector, var_offsets, var_sizes):
    """Converts a vector to a map of variable id to value.

    Parameters
    ----------
    vector : NumPy matrix
        The vector of values.
    var_offsets : dict
        A map of variable id to offset in the vector.
    var_sizes : dict
        A map of variable id to variable size.

    Returns
    -------
    dict
        A map of variable id to variable value.
    """
    val_dict = {}
    for id_, offset in var_offsets.items():
        size = var_sizes[id_]
        value = np.zeros(size)
        offset = var_offsets[id_]
        for col in range(size[1]):
            value[:, col] = vector[offset:size[0] + offset]
            offset += size[0]
        val_dict[id_] = value
    return val_dict