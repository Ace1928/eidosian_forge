from ... import initializer
from ..block import HybridBlock
from ...util import is_np_array
class Swish(HybridBlock):
    """
    Swish Activation function
        https://arxiv.org/pdf/1710.05941.pdf

    Parameters
    ----------
    beta : float
        swish(x) = x * sigmoid(beta*x)


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """

    def __init__(self, beta=1.0, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self._beta = beta

    def hybrid_forward(self, F, x):
        if is_np_array():
            return x * F.npx.sigmoid(self._beta * x)
        else:
            return x * F.sigmoid(self._beta * x, name='fwd')