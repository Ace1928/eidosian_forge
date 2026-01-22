from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import gen_training_ops
Optimizer that implements the Adamax algorithm.

  It is a variant of Adam based on the infinity norm.
  Default parameters follow those provided in the paper.
  Adamax is sometimes superior to adam, specially in models with embeddings.

  Initialization:

  ```python
  m = 0  # Initialize initial 1st moment vector
  v = 0  # Initialize the exponentially weighted infinity norm
  t = 0  # Initialize timestep
  ```

  The update rule for parameter `w` with gradient `g` is
  described at the end of section 7.1 of the paper:

  ```python
  t += 1
  m = beta1 * m + (1 - beta) * g
  v = max(beta2 * v, abs(g))
  current_lr = learning_rate / (1 - beta1 ** t)
  w = w - current_lr * m / (v + epsilon)
  ```

  Similarly to `Adam`, the epsilon is added for numerical stability
  (especially to get rid of division by zero when `v_t == 0`).

  In contrast to `Adam`, the sparse implementation of this algorithm
  (used when the gradient is an IndexedSlices object, typically because of
  `tf.gather` or an embedding lookup in the forward pass) only updates
  variable slices and corresponding `m_t`, `v_t` terms when that part of
  the variable was used in the forward pass. This means that the sparse
  behavior is contrast to the dense behavior (similar to some momentum
  implementations which ignore momentum unless a variable slice was actually
  used).

  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate.
    beta_1: A float value or a constant float tensor. The exponential decay
      rate for the 1st moment estimates.
    beta_2: A float value or a constant float tensor. The exponential decay
      rate for the exponentially weighted infinity norm.
    epsilon: A small constant for numerical stability.
    name: Optional name for the operations created when applying gradients.
      Defaults to `"Adamax"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Reference:
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
  