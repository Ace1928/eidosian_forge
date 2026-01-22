import collections
import copy
import multiprocessing.dummy
import multiprocessing.pool
import threading
import numpy as np
import six
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
class AllReduceCrossDeviceOps(CrossDeviceOps):
    """All-reduce implementation of CrossDeviceOps.

  It performs all-reduce when applicable using NCCL or hierarchical copy. For
  the batch API, tensors will be repacked or aggregated for more efficient
  cross-device transportation.

  For reduces that are not all-reduce, it falls back to
  `tf.distribute.ReductionToOneDevice`.
  """

    def __init__(self, all_reduce_alg='nccl', num_packs=1):
        """Initializes the object.

    Args:
      all_reduce_alg: the all-reduce algorithm to use, currently only "nccl" or
        "hierarchical_copy" are supported.
      num_packs: a non-negative integer. The number of packs to split values
        into. If zero, no packing will be done.
    """
        self._all_reduce_alg = all_reduce_alg
        self._num_packs = num_packs
        self._simple_cross_replica_ops = ReductionToOneDevice()
        super(AllReduceCrossDeviceOps, self).__init__()

    def reduce_implementation(self, reduce_op, per_replica_value, destinations, options):
        del options
        if _devices_match(per_replica_value, destinations) and (not any(('cpu' in d.lower() for d in get_devices_from(destinations)))):
            return self._batch_all_reduce(reduce_op, [per_replica_value])[0]
        else:
            return self._simple_cross_replica_ops.reduce(reduce_op, per_replica_value, destinations)

    def batch_reduce_implementation(self, reduce_op, value_destination_pairs, options):
        if _all_devices_match(value_destination_pairs):
            return self._batch_all_reduce(reduce_op, [v[0] for v in value_destination_pairs])
        else:
            return [self.reduce_implementation(reduce_op, value, dest, options) for value, dest in value_destination_pairs]

    def _batch_all_reduce(self, reduce_op, per_replica_values):
        """All-reduce algorithm in a batch."""
        dense_values, dense_indices, sparse_values, sparse_indices = cross_device_utils.split_by_sparsity(per_replica_values)
        if dense_values:
            dense_results = self._do_batch_all_reduce(reduce_op, dense_values)
        else:
            dense_results = []
        if sparse_values:
            sparse_results = self._do_batch_all_reduce_sparse(reduce_op, sparse_values)
        else:
            sparse_results = []
        return cross_device_utils.stitch_values(((dense_results, dense_indices), (sparse_results, sparse_indices)))

    def _do_batch_all_reduce(self, reduce_op, dense_values):
        """Run batch all-reduces."""
        logging.log_first_n(logging.INFO, 'batch_all_reduce: %d all-reduces with algorithm = %s, num_packs = %d' % (len(dense_values), self._all_reduce_alg, self._num_packs), 10)
        destinations = dense_values[0]._devices
        grouped = _group_value_by_device(dense_values)
        device_grad_packs, tensor_packer = _pack_tensors(grouped, self._num_packs)
        if self._all_reduce_alg == 'nccl':
            reduced = cross_device_utils.aggregate_gradients_using_nccl(device_grad_packs)
        else:
            reduced = cross_device_utils.aggregate_gradients_using_hierarchical_copy(destinations, device_grad_packs)
        reduced = _unpack_tensors(reduced, tensor_packer)
        return _ungroup_and_make_mirrored(reduced, dense_values[0], reduce_op)

    def _do_batch_all_reduce_sparse(self, reduce_op, sparse_values):
        """Run batch all-reduce for sparse values."""
        logging.log_first_n(logging.WARN, 'Efficient allreduce is not supported for %d IndexedSlices' % len(sparse_values), 10)
        return self._simple_cross_replica_ops.batch_reduce(reduce_op, zip(sparse_values, sparse_values))

    def _gather_implementation(self, per_replica_value, destinations, axis, options):
        logging.log_first_n(logging.WARN, "gather/all_gather with NCCL or HierarchicalCopy is not supported. Falling back to gather on one device and then broadcast. We're working on a more efficient implementation.", 3)
        return ReductionToOneDevice()._gather(per_replica_value, destinations, axis, options)