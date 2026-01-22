import threading
from tensorboard.compat import tf2 as tf
class LazyTensorCreator:
    """Lazy auto-converting wrapper for a callable that returns a `tf.Tensor`.

    This class wraps an arbitrary callable that returns a `Tensor` so that it
    will be automatically converted to a `Tensor` by any logic that calls
    `tf.convert_to_tensor()`. This also memoizes the callable so that it is
    called at most once.

    The intended use of this class is to defer the construction of a `Tensor`
    (e.g. to avoid unnecessary wasted computation, or ensure any new ops are
    created in a context only available later on in execution), while remaining
    compatible with APIs that expect to be given an already materialized value
    that can be converted to a `Tensor`.

    This class is thread-safe.
    """

    def __init__(self, tensor_callable):
        """Initializes a LazyTensorCreator object.

        Args:
          tensor_callable: A callable that returns a `tf.Tensor`.
        """
        if not callable(tensor_callable):
            raise ValueError('Not a callable: %r' % tensor_callable)
        self._tensor_callable = tensor_callable
        self._tensor = None
        self._tensor_lock = threading.RLock()
        _register_conversion_function_once()

    def __call__(self):
        if self._tensor is None or self._tensor is _CALL_IN_PROGRESS_SENTINEL:
            with self._tensor_lock:
                if self._tensor is _CALL_IN_PROGRESS_SENTINEL:
                    raise RuntimeError('Cannot use LazyTensorCreator with reentrant callable')
                elif self._tensor is None:
                    self._tensor = _CALL_IN_PROGRESS_SENTINEL
                    self._tensor = self._tensor_callable()
        return self._tensor