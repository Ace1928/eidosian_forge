from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['distributions.RegisterKL'])
class RegisterKL:
    """Decorator to register a KL divergence implementation function.

  Usage:

  @distributions.RegisterKL(distributions.Normal, distributions.Normal)
  def _kl_normal_mvn(norm_a, norm_b):
    # Return KL(norm_a || norm_b)
  """

    @deprecation.deprecated('2019-01-01', 'The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.', warn_once=True)
    def __init__(self, dist_cls_a, dist_cls_b):
        """Initialize the KL registrar.

    Args:
      dist_cls_a: the class of the first argument of the KL divergence.
      dist_cls_b: the class of the second argument of the KL divergence.
    """
        self._key = (dist_cls_a, dist_cls_b)

    def __call__(self, kl_fn):
        """Perform the KL registration.

    Args:
      kl_fn: The function to use for the KL divergence.

    Returns:
      kl_fn

    Raises:
      TypeError: if kl_fn is not a callable.
      ValueError: if a KL divergence function has already been registered for
        the given argument classes.
    """
        if not callable(kl_fn):
            raise TypeError('kl_fn must be callable, received: %s' % kl_fn)
        if self._key in _DIVERGENCES:
            raise ValueError('KL(%s || %s) has already been registered to: %s' % (self._key[0].__name__, self._key[1].__name__, _DIVERGENCES[self._key]))
        _DIVERGENCES[self._key] = kl_fn
        return kl_fn