import tensorflow as tf
from keras.src import backend
from keras.src.backend.common import KerasVariable
from keras.src.optimizers import base_optimizer
def _distributed_tf_update_step(self, distribution, grads_and_vars, learning_rate):

    def apply_grad_to_update_var(var, grad):
        return self.update_step(grad, var, learning_rate)
    for grad, var in grads_and_vars:
        distribution.extended.update(var, apply_grad_to_update_var, args=(grad,), group=False)