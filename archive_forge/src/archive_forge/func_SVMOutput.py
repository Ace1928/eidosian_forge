from ._internal import NDArrayBase
from ..base import _Null
def SVMOutput(data=None, label=None, margin=_Null, regularization_coefficient=_Null, use_linear=_Null, out=None, name=None, **kwargs):
    """Computes support vector machine based transformation of the input.

    This tutorial demonstrates using SVM as output layer for classification instead of softmax:
    https://github.com/apache/mxnet/tree/v1.x/example/svm_mnist.



    Parameters
    ----------
    data : NDArray
        Input data for SVM transformation.
    label : NDArray
        Class label for the input data.
    margin : float, optional, default=1
        The loss function penalizes outputs that lie outside this margin. Default margin is 1.
    regularization_coefficient : float, optional, default=1
        Regularization parameter for the SVM. This balances the tradeoff between coefficient size and error.
    use_linear : boolean, optional, default=0
        Whether to use L1-SVM objective. L2-SVM objective is used by default.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)