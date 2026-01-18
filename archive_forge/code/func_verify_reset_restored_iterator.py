import os
import numpy as np
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.data.experimental.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import nest
def verify_reset_restored_iterator(self, ds_fn, num_outputs, break_point=None, sparse_tensors=False, verify_exhausted=True):
    """Attempts to re-initialize a restored iterator.

    This is useful when restoring a training checkpoint during validation.

    Args:
      ds_fn: 0-argument function that returns a Dataset.
      num_outputs: Total number of outputs expected from this Dataset.
      break_point: Break point. Optional. Defaults to num_outputs/2.
      sparse_tensors: Whether dataset is built from SparseTensor(s).
      verify_exhausted: Whether to verify that the iterator has been exhausted
        after producing `num_outputs` elements.

    Raises:
      AssertionError if any test fails.
    """
    if context.executing_eagerly():
        self.skipTest('Eager mode iteration do not support re-initialization.')
    break_point = num_outputs // 2 if not break_point else break_point
    expected = self.gen_outputs(ds_fn, [], num_outputs, sparse_tensors=sparse_tensors, verify_exhausted=verify_exhausted)
    self.gen_outputs(ds_fn, [], break_point, sparse_tensors=sparse_tensors, verify_exhausted=False)
    actual = []
    with ops.Graph().as_default() as g:
        saver = self._import_meta_graph()
        init_op, get_next_op = self._get_iterator_ops_from_collection(ds_fn, sparse_tensors=sparse_tensors)
        get_next_op = remove_variants(get_next_op)
        with self.session(graph=g) as sess:
            self._initialize(init_op, sess)
            self._restore(saver, sess)
            self._initialize(init_op, sess)
            for _ in range(num_outputs):
                actual.append(sess.run(get_next_op))
            if verify_exhausted:
                with self.assertRaises(errors.OutOfRangeError):
                    sess.run(get_next_op)
    self.match(expected, actual)