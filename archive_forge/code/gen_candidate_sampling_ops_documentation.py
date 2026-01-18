import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
Generates labels for candidate sampling with a uniform distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).

    sampled_candidates: A `Tensor` of type `int64`.
    true_expected_count: A `Tensor` of type `float32`.
    sampled_expected_count: A `Tensor` of type `float32`.
  