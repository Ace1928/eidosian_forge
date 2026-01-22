from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import gen_training_ops
class NonFusedAdam(optimizer_v2.OptimizerV2):
    """Optimizer that implements the Adam algorithm without fused kernels.

  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.
  According to the paper
  [Adam: A Method for Stochastic Optimization. Kingma et al.,
  2014](http://arxiv.org/abs/1412.6980), the method is "*computationally
  efficient, has little memory requirement, invariant to diagonal rescaling of
  gradients, and is well suited for problems that are large in terms of
  data/parameters*".

  For AMSGrad see [On The Convergence Of Adam And Beyond.
  Reddi et al., 5-8](https://openreview.net/pdf?id=ryQu7f-RZ).

  **If amsgrad = False**:

  initialize $m_0$ as 1st moment vector
  initialize $v_0$ as 2nd moment vector

  The update rule for $\\theta$ with gradient $g$ uses an optimization
  described at the end of section 2 of the paper:

  $$lr_t = \\mathrm{learning\\_rate} *
    \\sqrt{1 - \\beta_2^t} / (1 - \\beta_1^t)$$
  $$m_t = \\beta_1 * m_{t-1} + (1 - \\beta_1) * g$$
  $$v_t = \\beta_2 * v_{t-1} + (1 - \\beta_2) * g^2$$
  $$\\theta_t = \\theta_{t-1} - lr_t * m_t / (\\sqrt{v_t} + \\epsilon)$$

  **If amsgrad = True**:

  initialize $m_0$ as 1st moment vector
  initialize $v_0$ as 2nd moment vector
  initialize $\\hat{v}_0$ as 2nd moment vector

  The update rule for $\\theta$ with gradient $g$ uses an optimization
  described at the end of section 2 of the paper:

  $$lr_t = \\mathrm{learning\\_rate} *
    \\sqrt{1 - \\beta_2^t} / (1 - \\beta_1^t)$$

  $$m_t = \\beta_1 * m_{t-1} + (1 - \\beta_1) * g$$
  $$v_t = \\beta_2 * v_{t-1} + (1 - \\beta_2) * g^2$$
  $$\\hat{v}_t = \\max(\\hat{v}_{t-1}, v_t)$$
  $$\\theta_t = \\theta_{t-1} - lr_t * m_t / (\\sqrt{\\hat{v}_t} + \\epsilon)$$

  The default value of 1e-7 for epsilon might not be a good default in
  general. For example, when training an Inception network on ImageNet a
  current good choice is 1.0 or 0.1. Note that since Adam uses the
  formulation just before Section 2.1 of the Kingma and Ba paper rather than
  the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
  hat" in the paper.

  The sparse implementation of this algorithm (used when the gradient is an
  IndexedSlices object, typically because of `tf.gather` or an embedding
  lookup in the forward pass) does apply momentum to variable slices even if
  they were not used in the forward pass (meaning they have a gradient equal
  to zero). Momentum decay (beta1) is also applied to the entire momentum
  accumulator. This means that the sparse behavior is equivalent to the dense
  behavior (in contrast to some momentum implementations which ignore momentum
  unless a variable slice was actually used).

  Usage:

  >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> # The first step is `-learning_rate*sign(grad)`
  >>> var1.numpy()
  9.9
  """
    _HAS_AGGREGATE_GRAD = True

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam', **kwargs):
        """Construct a new Adam optimizer.

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable that
        takes no arguments and returns the actual value to use, The learning
        rate. Defaults to 0.001.
      beta_1: A float value or a constant float tensor, or a callable that takes
        no arguments and returns the actual value to use. The exponential decay
        rate for the 1st moment estimates. Defaults to 0.9.
      beta_2: A float value or a constant float tensor, or a callable that takes
        no arguments and returns the actual value to use, The exponential decay
        rate for the 2nd moment estimates. Defaults to 0.999.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
      amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond". Defaults to `False`.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
        super(NonFusedAdam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(NonFusedAdam, self)._prepare_local(var_device, var_dtype, apply_state)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = apply_state[var_device, var_dtype]['lr_t'] * (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))
        apply_state[var_device, var_dtype].update(dict(lr=lr, epsilon=tensor_conversion.convert_to_tensor_v2_with_dispatch(self.epsilon, var_dtype), beta_1_t=beta_1_t, beta_1_power=beta_1_power, one_minus_beta_1_t=1 - beta_1_t, beta_2_t=beta_2_t, beta_2_power=beta_2_power, one_minus_beta_2_t=1 - beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(NonFusedAdam, self).set_weights(weights)

    @def_function.function(jit_compile=True)
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = (var.device, var.dtype.base_dtype)
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        alpha = coefficients['lr_t'] * math_ops.sqrt(1 - coefficients['beta_2_power']) / (1 - coefficients['beta_1_power'])
        m.assign_add((grad - m) * (1 - coefficients['beta_1_t']))
        v.assign_add((math_ops.square(grad) - v) * (1 - coefficients['beta_2_t']))
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat.assign(math_ops.maximum(vhat, v))
            v = vhat
        var.assign_sub(m * alpha / (math_ops.sqrt(v) - coefficients['epsilon']))

    @def_function.function(jit_compile=True)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = (var.device, var.dtype.base_dtype)
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m.assign(m * coefficients['beta_1_t'])
        m.scatter_add(indexed_slices.IndexedSlices(m_scaled_g_values, indices))
        v = self.get_slot(var, 'v')
        v_scaled_g_values = grad * grad * coefficients['one_minus_beta_2_t']
        v.assign(v * coefficients['beta_2_t'])
        v.scatter_add(indexed_slices.IndexedSlices(v_scaled_g_values, indices))
        if not self.amsgrad:
            var.assign_sub(coefficients['lr'] * m / (math_ops.sqrt(v) + coefficients['epsilon']))
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat.assign(math_ops.maximum(v_hat, v))
            var.assign_sub(coefficients['lr'] * m / (math_ops.sqrt(v_hat) + coefficients['epsilon']))

    def get_config(self):
        config = super(NonFusedAdam, self).get_config()
        config.update({'learning_rate': self._serialize_hyperparameter('learning_rate'), 'decay': self._initial_decay, 'beta_1': self._serialize_hyperparameter('beta_1'), 'beta_2': self._serialize_hyperparameter('beta_2'), 'epsilon': self.epsilon, 'amsgrad': self.amsgrad})
        return config