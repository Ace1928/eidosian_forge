import warnings
from string import ascii_letters as ABC
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
def _process_jacs(jac, qhess):
    """
    Combine the classical and quantum jacobians
    """
    if not qml.math.is_abstract(jac):
        shape = qml.math.shape(jac)
        is_square = len(shape) == 2 and shape[0] == shape[1]
        if is_square and qml.math.allclose(jac, qml.numpy.eye(shape[0])):
            return qhess if len(qhess) > 1 else qhess[0]
    hess = []
    for qh in qhess:
        if not isinstance(qh, tuple) or not isinstance(qh[0], tuple):
            qh = qml.math.expand_dims(qh, [0, 1])
        else:
            qh = qml.math.stack([qml.math.stack(row) for row in qh])
        jac_ndim = len(qml.math.shape(jac))
        qh_indices = 'ab...'
        first_jac_indices = f'a{ABC[2:2 + jac_ndim - 1]}'
        second_jac_indices = f'b{ABC[2 + jac_ndim - 1:2 + 2 * jac_ndim - 2]}'
        result_indices = f'{ABC[2:2 + 2 * jac_ndim - 2]}...'
        qh = qml.math.einsum(f'{qh_indices},{first_jac_indices},{second_jac_indices}->{result_indices}', qh, jac, jac)
        hess.append(qh)
    return tuple(hess) if len(hess) > 1 else hess[0]