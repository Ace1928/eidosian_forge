from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_batch_ops
from tensorflow.python.ops.gen_batch_ops import *
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
Batches the computation done by the decorated function.

  So, for example, in the following code

  ```python
  @batch_function(1, 2, 3)
  def layer(a):
    return tf.matmul(a, a)

  b = layer(w)
  ```

  if more than one session.run call is simultaneously trying to compute `b`
  the values of `w` will be gathered, non-deterministically concatenated
  along the first axis, and only one thread will run the computation. See the
  documentation of the `Batch` op for more details.

  Assumes that all arguments of the decorated function are Tensors which will
  be batched along their first dimension.

  SparseTensor is not supported. The return value of the decorated function
  must be a Tensor or a list/tuple of Tensors.

  Args:
    num_batch_threads: Number of scheduling threads for processing batches
     of work. Determines the number of batches processed in parallel.
    max_batch_size: Batch sizes will never be bigger than this.
    batch_timeout_micros: Maximum number of microseconds to wait before
     outputting an incomplete batch.
    allowed_batch_sizes: Optional list of allowed batch sizes. If left empty,
     does nothing. Otherwise, supplies a list of batch sizes, causing the op
     to pad batches up to one of those sizes. The entries must increase
     monotonically, and the final entry must equal max_batch_size.
    max_enqueued_batches: The maximum depth of the batch queue. Defaults to 10.
    autograph: Whether to use autograph to compile python and eager style code
     for efficient graph-mode execution.
    enable_large_batch_splitting: The value of this option doesn't affect
     processing output given the same input; it affects implementation details
     as stated below: 1. Improve batching efficiency by eliminating unnecessary
     adding. 2.`max_batch_size` specifies the limit of input and
     `allowed_batch_sizes` specifies the limit of a task to be processed. API
     user can give an input of size 128 when 'max_execution_batch_size'
     is 32 -> implementation can split input of 128 into 4 x 32, schedule
     concurrent processing, and then return concatenated results corresponding
     to 128.

  Returns:
    The decorated function will return the unbatched computation output Tensors.
  