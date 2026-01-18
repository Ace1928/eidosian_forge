import threading
from tensorboard.compat import tf2 as tf
Initializes a LazyTensorCreator object.

        Args:
          tensor_callable: A callable that returns a `tf.Tensor`.
        