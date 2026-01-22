import warnings
import numpy
import cupy
import cupy.linalg as linalg
from cupyx.scipy.sparse import linalg as splinalg
def _makeOperator(operatorInput, expectedShape):
    """Takes a dense numpy array or a sparse matrix or
    a function and makes an operator performing matrix * blockvector
    products.
    """
    if operatorInput is None:
        return None
    else:
        operator = splinalg.aslinearoperator(operatorInput)
    if operator.shape != expectedShape:
        raise ValueError('operator has invalid shape')
    return operator