from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_candidate_sampling_ops
from tensorflow.python.ops import math_ops  # pylint: disable=unused-import
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('random.all_candidate_sampler', 'nn.all_candidate_sampler')
def all_candidate_sampler(true_classes, num_true, num_sampled, unique, seed=None, name=None):
    """Generate the set of all classes.

  Deterministically generates and returns the set of all possible classes.
  For testing purposes.  There is no need to use this, since you might as
  well use full softmax or full logistic regression.

  Args:
    true_classes: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
    num_true: An `int`.  The number of target classes per training example.
    num_sampled: An `int`.  The number of possible classes.
    unique: A `bool`. Ignored.
      unique.
    seed: An `int`. An operation-specific seed. Default is 0.
    name: A name for the operation (optional).

  Returns:
    sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
      This operation deterministically returns the entire range
      `[0, num_sampled]`.
    true_expected_count: A tensor of type `float`.  Same shape as
      `true_classes`. The expected counts under the sampling distribution
      of each of `true_classes`. All returned values are 1.0.
    sampled_expected_count: A tensor of type `float`. Same shape as
      `sampled_candidates`. The expected counts under the sampling distribution
      of each of `sampled_candidates`. All returned values are 1.0.
  """
    seed1, seed2 = random_seed.get_seed(seed)
    return gen_candidate_sampling_ops.all_candidate_sampler(true_classes, num_true, num_sampled, unique, seed=seed1, seed2=seed2, name=name)