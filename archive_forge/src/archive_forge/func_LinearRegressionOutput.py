from ._internal import NDArrayBase
from ..base import _Null
def LinearRegressionOutput(data=None, label=None, grad_scale=_Null, out=None, name=None, **kwargs):
    """Computes and optimizes for squared loss during backward propagation.
    Just outputs ``data`` during forward propagation.

    If :math:`\\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
    then the squared loss estimated over :math:`n` samples is defined as

    :math:`\\text{SquaredLoss}(\\textbf{Y}, \\hat{\\textbf{Y}} ) = \\frac{1}{n} \\sum_{i=0}^{n-1} \\lVert  \\textbf{y}_i - \\hat{\\textbf{y}}_i  \\rVert_2`

    .. note::
       Use the LinearRegressionOutput as the final output layer of a net.

    The storage type of ``label`` can be ``default`` or ``csr``

    - LinearRegressionOutput(default, default) = default
    - LinearRegressionOutput(default, csr) = default

    By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
    The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.



    Defined in ../src/operator/regression_output.cc:L92

    Parameters
    ----------
    data : NDArray
        Input data to the function.
    label : NDArray
        Input label to the function.
    grad_scale : float, optional, default=1
        Scale the gradient by a float factor

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)