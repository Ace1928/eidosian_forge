from ._internal import NDArrayBase
from ..base import _Null
def LogisticRegressionOutput(data=None, label=None, grad_scale=_Null, out=None, name=None, **kwargs):
    """Applies a logistic function to the input.

    The logistic function, also known as the sigmoid function, is computed as
    :math:`\\frac{1}{1+exp(-\\textbf{x})}`.

    Commonly, the sigmoid is used to squash the real-valued output of a linear model
    :math:`wTx+b` into the [0,1] range so that it can be interpreted as a probability.
    It is suitable for binary classification or probability prediction tasks.

    .. note::
       Use the LogisticRegressionOutput as the final output layer of a net.

    The storage type of ``label`` can be ``default`` or ``csr``

    - LogisticRegressionOutput(default, default) = default
    - LogisticRegressionOutput(default, csr) = default

    The loss function used is the Binary Cross Entropy Loss:

    :math:`-{(y\\log(p) + (1 - y)\\log(1 - p))}`

    Where `y` is the ground truth probability of positive outcome for a given example, and `p` the probability predicted by the model. By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
    The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.



    Defined in ../src/operator/regression_output.cc:L152

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