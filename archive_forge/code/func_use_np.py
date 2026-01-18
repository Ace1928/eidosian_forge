import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def use_np(func):
    """A convenience decorator for wrapping user provided functions and classes in the scope of
    both NumPy-shape and NumPy-array semantics, which means that (1) empty tuples `()` and tuples
    with zeros, such as `(0, 1)`, `(1, 0, 2)`, will be treated as scalar tensors' shapes and
    zero-size tensors' shapes in shape inference functions of operators, instead of as unknown
    in legacy mode; (2) ndarrays of type `mxnet.numpy.ndarray` should be created instead of
    `mx.nd.NDArray`.

    Example::
        import mxnet as mx
        from mxnet import gluon, np


        class TestHybridBlock1(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock1, self).__init__()
                self.w = self.params.get('w', shape=(2, 2))

            def hybrid_forward(self, F, x, w):
                return F.dot(x, w) + F.ones((1,))


        x = mx.nd.ones((2, 2))
        net1 = TestHybridBlock1()
        net1.initialize()
        out = net1.forward(x)
        for _, v in net1.collect_params().items():
            assert type(v.data()) is mx.nd.NDArray
        assert type(out) is mx.nd.NDArray


        @np.use_np
        class TestHybridBlock2(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock2, self).__init__()
                self.w = self.params.get('w', shape=(2, 2))

            def hybrid_forward(self, F, x, w):
                return F.np.dot(x, w) + F.np.ones(())


        x = np.ones((2, 2))
        net2 = TestHybridBlock2()
        net2.initialize()
        out = net2.forward(x)
        for _, v in net2.collect_params().items():
            print(type(v.data()))
            assert type(v.data()) is np.ndarray
        assert type(out) is np.ndarray

    Parameters
    ----------
    func : a user-provided callable function or class to be scoped by the
    NumPy-shape and NumPy-array semantics.

    Returns
    -------
    Function or class
        A function or class wrapped in the Numpy-shape and NumPy-array scope.
    """
    return use_np_shape(use_np_array(func))