import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
class CollectiveReplicaLauncher(object):
    """Launch collectives on one replica."""
    _prefer_unique_instance_key = True
    _prefer_ordering_token = True

    def __init__(self, group_key: int, group_size: int, collective_keys: CollectiveKeys, device: str, options: collective_util.Options):
        self._group_key = group_key
        self._group_size = group_size
        self._collective_keys = collective_keys
        self._device = device
        self._options = options
        if self._use_ordering_token():
            with ops.init_scope(), ops.device(device):
                self._ordering_token = resource_variable_ops.ResourceVariable(0.0)
        else:
            self._ordering_token = None

    def _control_input(self, control_input: Union[core.TensorLike, ops.Operation]):
        if control_input is not None and (not self._use_ordering_token()):
            return ops.control_dependencies([control_input])
        return ops.NullContextmanager()

    def _use_unique_instance_key(self):
        if not ops.executing_eagerly_outside_functions():
            return False
        return CollectiveReplicaLauncher._prefer_unique_instance_key

    def _use_ordering_token(self):
        if not ops.executing_eagerly_outside_functions():
            return False
        return CollectiveReplicaLauncher._prefer_ordering_token

    def _next_instance_key(self):
        """Returns the next instance key."""
        if self._use_unique_instance_key():
            graph = ops.get_default_graph()
            while getattr(graph, 'is_control_flow_graph', False):
                graph = graph.outer_graph
            if not context.executing_eagerly() and graph.building_function:
                with graph.as_default():
                    return graph.capture_call_time_value(self._next_instance_key, tensor_spec.TensorSpec([], dtypes.int32))
            else:
                instance_key = self._collective_keys.get_instance_key(self._group_key, self._device)
                with ops.device('CPU:0'):
                    return ops.convert_to_tensor(instance_key, dtype=dtypes.int32)
        else:
            return self._collective_keys.get_instance_key(self._group_key, self._device)

    def _get_ordering_token(self):
        if self._use_ordering_token():
            return self._ordering_token.handle

    def can_order_nccl(self):
        """Whether this launcher can order NCCL operations."""
        return self._use_ordering_token()

    def all_reduce(self, input_tensor: core.TensorLike, control_input: Optional[Union[core.TensorLike, ops.Operation]]=None, options: Optional[collective_util.Options]=None) -> core.Tensor:
        """All-reduce a dense tensor.

    Args:
      input_tensor: a dense tensor. It must have the same shape on all replicas.
      control_input: if not None, add control edges between control_input and
        the all-reduce.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The reduced tensor.
    """
        instance_key = self._next_instance_key()
        options = self._options.merge(options)
        ordering_token = self._get_ordering_token()
        with ops.device(self._device), self._control_input(control_input):
            return collective_ops.all_reduce_v2(input_tensor, self._group_size, self._group_key, instance_key, communication_hint=options.implementation.value, timeout=options.timeout_seconds, ordering_token=ordering_token)

    def _all_gather(self, input_tensor: core.TensorLike, options: Optional[collective_util.Options]) -> core.Tensor:
        """All-gather a dense tensor.

    Args:
      input_tensor: a dense tensor. It must have the same shape on all replicas.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The reduced tensor.
    """
        instance_key = self._next_instance_key()
        options = self._options.merge(options)
        ordering_token = self._get_ordering_token()
        with ops.device(self._device):
            return collective_ops.all_gather_v2(input_tensor, self._group_size, self._group_key, instance_key, communication_hint=options.implementation.value, timeout=options.timeout_seconds, ordering_token=ordering_token)

    def batch_all_reduce(self, input_tensor_packs: List[List[core.TensorLike]], options: Optional[collective_util.Options]=None) -> core.Tensor:
        """Batch all-reduce dense tensors.

    This takes a list of batches of tensors. Using multiple batches have the
    benefit that it doesn't need to wait for all inputs to be ready to start the
    all-reduce.

    Args:
      input_tensor_packs: a list of lists of dense tensors.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      A flat list of reduced tensors.
    """
        options = self._options.merge(options)
        outputs = []
        for pack in input_tensor_packs:
            if context.executing_eagerly():
                for input_tensor in pack:
                    outputs.append(self.all_reduce(input_tensor, None, options))
            else:
                with ops.device(self._device):
                    flat_tensors = [array_ops.reshape(t, [-1]) for t in pack]
                    shapes = [array_ops.shape(t) for t in pack]
                    if options.implementation == collective_util.CommunicationImplementation.NCCL and outputs:
                        control_input = outputs[-1]
                    else:
                        control_input = None
                    reduced = self.all_reduce(array_ops.concat(flat_tensors, axis=0), control_input, options)
                    num_elements = [math_ops.reduce_prod(s) for s in shapes]
                    flat_outputs = array_ops.split(reduced, num_elements, axis=0)
                    for shape, flat_output in zip(shapes, flat_outputs):
                        outputs.append(array_ops.reshape(flat_output, shape))
        return outputs

    def all_gather(self, input_tensor: core.TensorLike, axis: core.TensorLike, options: Optional[collective_util.Options]=None) -> core.Tensor:
        """All-gather a dense tensor.

    This method must be called inside a tf.function.

    Args:
      input_tensor: a dense tensor. It must have the same rank on all replicas,
        and dimensions other than `axis` need to be the same as well.
      axis: 0-D int32 Tensor. Dimension along which to gather. Must be in the
        range [0, rank(value)).
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The gathered Tensor.

    Raises:
      RuntimeError: if called in eager mode.
    """
        if context.executing_eagerly():
            raise RuntimeError('all_gather is not supported in eager mode.')
        with ops.device(self._device), ops.control_dependencies([array_ops.identity(input_tensor)]):
            perm_pre = array_ops.concat(([axis], math_ops.range(axis), math_ops.range(axis + 1, array_ops.rank(input_tensor))), axis=0)
            input_tensor_t = array_ops.transpose(input_tensor, perm=perm_pre)
            gathered_shape = self._all_gather(array_ops.expand_dims_v2(array_ops.shape_v2(input_tensor_t), axis=0), options)
            first_dims = gathered_shape[:, 0]
            full_axis_dim = math_ops.reduce_max(first_dims)
            padded_input_tensor = _pad_util(input_tensor_t, full_axis_dim)
            gather_padded_out_tensor = self._all_gather(padded_input_tensor, options)
            split_tensors = []
            for i in range(self._group_size):
                start_pos = i * full_axis_dim
                split_tensors.append(gather_padded_out_tensor[start_pos:start_pos + first_dims[i]])
            out_tensor_t = array_ops.concat(split_tensors, 0)
            perm_after = array_ops.concat((math_ops.range(1, axis + 1), [0], math_ops.range(axis + 1, array_ops.rank(input_tensor_t))), axis=0)
            return array_ops.transpose(out_tensor_t, perm=perm_after)

    def all_reduce_indexed_slices(self, input_slices: indexed_slices.IndexedSlices, options: Optional[collective_util.Options]=None) -> indexed_slices.IndexedSlices:
        """All-reduce an IndexedSlices.

    This method can be called outside  tf.function.

    Args:
      input_slices: an IndexedSlices.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The reduced IndexedSlices.
    """
        options = self._options.merge(options)
        with ops.device(self._device):

            def all_gather_indexed_slices(all_gather_fn: Callable[[core.TensorLike, Optional[collective_util.Options]], core.Tensor]) -> indexed_slices.IndexedSlices:
                """Use all_gather_fn to aggregate `IndexedSlices`."""
                all_values = all_gather_fn(input_slices.values, options)
                if options.implementation == collective_util.CommunicationImplementation.NCCL:
                    control = [all_values]
                else:
                    control = []
                with ops.control_dependencies(control):
                    all_indices = all_gather_fn(input_slices.indices, options)
                return indexed_slices.IndexedSlices(values=all_values, indices=all_indices, dense_shape=input_slices.dense_shape)
            length = array_ops.shape(input_slices.indices)
            all_lengths = self._all_gather(length, options)

            def all_gather_with_padding(input_tensor: core.TensorLike, options: Optional[collective_util.Options]) -> core.Tensor:
                """all_gather tensors of different sizes using padding."""
                max_length = math_ops.reduce_max(all_lengths)
                padded_tensor = _pad_util(input_tensor, max_length)
                all_padded_tensors = self._all_gather(padded_tensor, options)
                split_tensors = []
                for i in range(self._group_size):
                    start_pos = i * max_length
                    split_tensors.append(all_padded_tensors[start_pos:start_pos + all_lengths[i]])
                return array_ops.concat(split_tensors, 0)
            return cond.cond(math_ops.equal(math_ops.reduce_max(all_lengths), math_ops.reduce_min(all_lengths)), lambda: all_gather_indexed_slices(self._all_gather), lambda: all_gather_indexed_slices(all_gather_with_padding))