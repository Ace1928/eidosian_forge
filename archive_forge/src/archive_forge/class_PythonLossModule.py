import logging
from .base_module import BaseModule
from ..initializer import Uniform
from .. import ndarray as nd
class PythonLossModule(PythonModule):
    """A convenient module class that implements many of the module APIs as
    empty functions.

    Parameters
    ----------
    name : str
        Names of the module. The outputs will be named `[name + '_output']`.
    data_names : list of str
        Defaults to ``['data']``. Names of the data expected by this module.
        Should be a list of only one name.
    label_names : list of str
        Default ``['softmax_label']``. Names of the labels expected by the module.
        Should be a list of only one name.
    grad_func : function
        Optional. If not ``None``, should be a function that takes `scores`
        and `labels`, both of type `NDArray`, and return the gradients with
        respect to the scores according to this loss function. The return
        value could be a numpy array or an `NDArray`.
    """

    def __init__(self, name='pyloss', data_names=('data',), label_names=('softmax_label',), logger=logging, grad_func=None):
        super(PythonLossModule, self).__init__(data_names, label_names, [name + '_output'], logger=logger)
        self._name = name
        assert len(data_names) == 1
        assert len(label_names) == 1
        self._scores = None
        self._labels = None
        self._scores_grad = None
        if grad_func is not None:
            assert callable(grad_func)
        self._grad_func = grad_func

    def _compute_output_shapes(self):
        """Computes the shapes of outputs. As a loss module with outputs, we simply
        output whatever we receive as inputs (i.e. the scores).
        """
        return [(self._name + '_output', self._data_shapes[0][1])]

    def forward(self, data_batch, is_train=None):
        """Forward computation. Here we do nothing but to keep a reference to
        the scores and the labels so that we can do backward computation.

        Parameters
        ----------
        data_batch : DataBatch
            Could be anything with similar API implemented.
        is_train : bool
            Default is ``None``, which means `is_train` takes the value of ``self.for_training``.
        """
        self._scores = data_batch.data[0]
        if is_train is None:
            is_train = self.for_training
        if is_train:
            self._labels = data_batch.label[0]

    def get_outputs(self, merge_multi_context=True):
        """Gets outputs of the previous forward computation. As a output loss module,
        we treat the inputs to this module as scores, and simply return them.

        Parameters
        ----------
        merge_multi_context : bool
            Should always be ``True``, because we do not use multiple contexts for computing.
        """
        assert merge_multi_context is True
        return [self._scores]

    def backward(self, out_grads=None):
        """Backward computation.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        """
        assert out_grads is None, 'For a loss module, out_grads should be None'
        assert self.for_training
        self._backward_impl()

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

    def get_input_grads(self, merge_multi_context=True):
        """Gets the gradients to the inputs, computed in the previous backward computation.

        Parameters
        ----------
        merge_multi_context : bool
            Should always be ``True`` because we do not use multiple context for computation.
        """
        assert merge_multi_context is True
        return [self._scores_grad]

    def install_monitor(self, mon):
        """Installs monitor on all executors."""
        raise NotImplementedError()