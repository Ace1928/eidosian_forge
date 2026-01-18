from ._internal import NDArrayBase
from ..base import _Null
Solves matrix equation involving a lower triangular matrix.
    Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
    on the leading *n-2* dimensions.

    If *n=2*, *A* must be triangular. The operator performs the BLAS3 function
    *trsm*, solving for *out* in:

       *op*\ (*A*) \* *out* = *alpha* \* *B*

    if *rightside=False*, or

       *out* \* *op*\ (*A*) = *alpha* \* *B*

    if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the
    identity or the matrix transposition (depending on *transpose*).

    If *n>2*, *trsm* is performed separately on the trailing two dimensions for all inputs
    (batch mode).

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

       Single matrix solve
       A = [[1.0, 0], [1.0, 1.0]]
       B = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
       trsm(A, B, alpha=0.5) = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

       Batch matrix solve
       A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
       B = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
            [[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]]
       trsm(A, B, alpha=0.5) = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]


    Defined in ../src/operator/tensor/la_op.cc:L395

    Parameters
    ----------
    A : NDArray
        Tensor of lower triangular matrices
    B : NDArray
        Tensor of matrices
    transpose : boolean, optional, default=0
        Use transposed of the triangular matrix
    rightside : boolean, optional, default=0
        Multiply triangular matrix from the right to non-triangular one.
    lower : boolean, optional, default=1
        True if the triangular matrix is lower triangular, false if it is upper triangular.
    alpha : double, optional, default=1
        Scalar factor to be applied to the result.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    