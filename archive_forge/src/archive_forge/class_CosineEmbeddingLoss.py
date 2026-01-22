import warnings
from .distance import PairwiseDistance
from .module import Module
from .. import functional as F
from .. import _reduction as _Reduction
from torch import Tensor
from typing import Callable, Optional
class CosineEmbeddingLoss(_Loss):
    """Creates a criterion that measures the loss given input tensors
    :math:`x_1`, :math:`x_2` and a `Tensor` label :math:`y` with values 1 or -1.
    Use (:math:`y=1`) to maximize the cosine similarity of two inputs, and (:math:`y=-1`) otherwise.
    This is typically used for learning nonlinear
    embeddings or semi-supervised learning.

    The loss function for each sample is:

    .. math::
        \\text{loss}(x, y) =
        \\begin{cases}
        1 - \\cos(x_1, x_2), & \\text{if } y = 1 \\\\
        \\max(0, \\cos(x_1, x_2) - \\text{margin}), & \\text{if } y = -1
        \\end{cases}

    Args:
        margin (float, optional): Should be a number from :math:`-1` to :math:`1`,
            :math:`0` to :math:`0.5` is suggested. If :attr:`margin` is missing, the
            default value is :math:`0`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input1: :math:`(N, D)` or :math:`(D)`, where `N` is the batch size and `D` is the embedding dimension.
        - Input2: :math:`(N, D)` or :math:`(D)`, same shape as Input1.
        - Target: :math:`(N)` or :math:`()`.
        - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`, otherwise scalar.

    Examples::

        >>> loss = nn.CosineEmbeddingLoss()
        >>> input1 = torch.randn(3, 5, requires_grad=True)
        >>> input2 = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.ones(3)
        >>> output = loss(input1, input2, target)
        >>> output.backward()
    """
    __constants__ = ['margin', 'reduction']
    margin: float

    def __init__(self, margin: float=0.0, size_average=None, reduce=None, reduction: str='mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.cosine_embedding_loss(input1, input2, target, margin=self.margin, reduction=self.reduction)