from tensorflow.python.framework import config
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import optimizer
from tensorflow.python.training.experimental import loss_scale_optimizer as loss_scale_optimizer_v1
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _wrap_optimizer(opt, loss_scale):
    """Wraps an optimizer with a LossScaleOptimizer."""
    for _, wrapper_optimizer in _REGISTERED_WRAPPER_OPTIMIZER_CLS.values():
        if isinstance(opt, wrapper_optimizer):
            raise ValueError('"opt" must not already be an instance of a {cls}. `enable_mixed_precision_graph_rewrite` will automatically wrap the optimizer with a {cls}.'.format(cls=wrapper_optimizer.__name__))
    for optimizer_cls, (wrapper_fn, _) in _REGISTERED_WRAPPER_OPTIMIZER_CLS.items():
        if isinstance(opt, optimizer_cls):
            return wrapper_fn(opt, loss_scale)
    raise ValueError('"opt" must be an instance of a tf.train.Optimizer or a tf.keras.optimizers.Optimizer, but got: %s' % opt)