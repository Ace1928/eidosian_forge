import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import optimizers
from keras.src.dtensor import utils as dtensor_utils
from keras.src.optimizers import optimizer
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.legacy import optimizer_v2
from keras.src.saving import serialization_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import keras_export
def _maybe_warn_about_scaling(loss_has_been_scaled, gradients_have_been_unscaled):
    """Warn if the loss or gradients hasn't been scaled or unscaled."""
    if loss_has_been_scaled and gradients_have_been_unscaled:
        return
    example_code = '\n    with tf.GradientTape() as tape:\n      loss = loss_fn()\n      scaled_loss = opt.get_scaled_loss(loss)\n    scaled_grads = tape.gradient(scaled_loss, vars)\n    grads = opt.get_unscaled_gradients(scaled_grads)\n    opt.apply_gradients([(grads, var)])'
    if not loss_has_been_scaled and (not gradients_have_been_unscaled):
        tf_logging.warning(f'You forgot to call LossScaleOptimizer.get_scaled_loss() and LossScaleOptimizer.get_unscaled_gradients() before calling LossScaleOptimizer.apply_gradients(). This will likely result in worse model quality, so please call them in the correct places! For example:{example_code}\nFor more information, see https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer')
    elif not loss_has_been_scaled:
        tf_logging.warning(f'You forgot to call LossScaleOptimizer.get_scaled_loss() before calling LossScaleOptimizer.apply_gradients() (you did call get_unscaled_gradients() however). This will likely result in worse model quality, so please call get_scaled_loss() in the correct place! For example:{example_code}\nFor more information, see https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer')
    elif not gradients_have_been_unscaled:
        tf_logging.warning(f'You forgot to call LossScaleOptimizer.get_unscaled_gradients() before calling LossScaleOptimizer.apply_gradients() (you did call get_scaled_loss() however). This will likely result in worse model quality, so please call get_unscaled_gradients() in the correct place! For example:{example_code}\nFor more information, see https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer')