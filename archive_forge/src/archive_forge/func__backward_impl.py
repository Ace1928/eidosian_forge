import logging
from .base_module import BaseModule
from ..initializer import Uniform
from .. import ndarray as nd
def _backward_impl(self):
    """Actual implementation of the backward computation. The computation
        should take ``self._scores`` and ``self._labels`` and then compute the
        gradients with respect to the scores, store it as an `NDArray` in
        ``self._scores_grad``.

        Instead of defining a subclass and overriding this function,
        a more convenient way is to pass in a `grad_func` when constructing
        the module object. Then it will be called to compute the gradients.
        """
    if self._grad_func is not None:
        grad = self._grad_func(self._scores, self._labels)
        if not isinstance(grad, nd.NDArray):
            grad = nd.array(grad)
        self._scores_grad = grad
    else:
        raise NotImplementedError()