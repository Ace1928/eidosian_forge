import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
Applies noisy linear cosine decay to the learning rate.

  Note that linear cosine decay is more aggressive than cosine decay and
  larger initial learning rates can typically be used.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies a noisy linear
  cosine decay function to a provided initial learning rate.
  It requires a `global_step` value to compute the decayed learning rate.
  You can just pass a TensorFlow variable that you increment at each
  training step.

  The function returns the decayed learning rate.  It is computed as:
  ```python
  global_step = min(global_step, decay_steps)
  linear_decay = (decay_steps - global_step) / decay_steps)
  cosine_decay = 0.5 * (
      1 + cos(pi * 2 * num_periods * global_step / decay_steps))
  decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
  decayed_learning_rate = learning_rate * decayed
  ```
  where eps_t is 0-centered gaussian noise with variance
  initial_variance / (1 + global_step) ** variance_decay

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed = noisy_linear_cosine_decay(
    learning_rate, global_step, decay_steps)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Number
      of steps to decay over.
    initial_variance: initial variance for the noise. See computation above.
    variance_decay: decay for the noise's variance. See computation above.
    num_periods: Number of periods in the cosine part of the decay. See
      computation above.
    alpha: See computation above.
    beta: See computation above.
    name: String.  Optional name of the operation.  Defaults to
      'NoisyLinearCosineDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.

  References:
    Neural Optimizer Search with Reinforcement Learning:
      [Bello et al., 2017](http://proceedings.mlr.press/v70/bello17a.html)
      ([pdf](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf))
    Stochastic Gradient Descent with Warm Restarts:
      [Loshchilov et al., 2017]
      (https://openreview.net/forum?id=Skq89Scxx&noteId=Skq89Scxx)
      ([pdf](https://openreview.net/pdf?id=Skq89Scxx))

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  