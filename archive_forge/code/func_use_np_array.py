import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def use_np_array(func):
    """A decorator wrapping Gluon `Block`s and all its methods, properties, and static functions
    with the semantics of NumPy-array, which means that where ndarrays are created,
    `mxnet.numpy.ndarray`s should be created, instead of legacy ndarrays of type `mx.nd.NDArray`.
    For example, at the time when a parameter is created in a `Block`, an `mxnet.numpy.ndarray`
    is created if it's decorated with this decorator.

    Example::
        import mxnet as mx
        from mxnet import gluon, np


        class TestHybridBlock1(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock1, self).__init__()
                self.w = self.params.get('w', shape=(2, 2))

            def hybrid_forward(self, F, x, w):
                return F.dot(x, w)


        x = mx.nd.ones((2, 2))
        net1 = TestHybridBlock1()
        net1.initialize()
        out = net1.forward(x)
        for _, v in net1.collect_params().items():
            assert type(v.data()) is mx.nd.NDArray
        assert type(out) is mx.nd.NDArray


        @np.use_np_array
        class TestHybridBlock2(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock2, self).__init__()
                self.w = self.params.get('w', shape=(2, 2))

            def hybrid_forward(self, F, x, w):
                return F.np.dot(x, w)


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
    func : a user-provided callable function or class to be scoped by the NumPy-array semantics.

    Returns
    -------
    Function or class
        A function or class wrapped in the NumPy-array scope.
    """
    if inspect.isclass(func):
        for name, method in inspect.getmembers(func, predicate=lambda f: inspect.isfunction(f) or inspect.ismethod(f) or isinstance(f, property)):
            if isinstance(method, property):
                setattr(func, name, property(use_np_array(method.__get__), method.__set__, method.__delattr__, method.__doc__))
            else:
                setattr(func, name, use_np_array(method))
        return func
    elif callable(func):

        @functools.wraps(func)
        def _with_np_array(*args, **kwargs):
            with np_array(active=True):
                return func(*args, **kwargs)
        return _with_np_array
    else:
        raise TypeError('use_np_array can only decorate classes and callable objects, while received a {}'.format(str(type(func))))