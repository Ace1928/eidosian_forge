import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
class Sharding(object):
    """A class to support adding sharding attributes to Ops.

  Use the factory constructors and then call apply_to_tensor:
    Sharding.replicate().apply_to_tensor(tensor)
  """

    def __init__(self, proto=None):
        """Do not use this constructor; use the factory functions below."""
        self._proto = proto

    @classmethod
    def replicate(cls):
        """Returns a replicated sharding attribute.

    This causes an op to be computed in its entirety independently on all
    cores in the XLA device.
    """
        return Sharding(proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED))

    @classmethod
    def manual(cls):
        """Returns a manuall sharding attribute.

    This means the op is manually partitioned by the user and XLA will not
    change the shapes.
    """
        return Sharding(proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.MANUAL))

    @classmethod
    def assign_device(cls, core):
        """Returns an AssignDevice sharding attribute.

    This causes an op to be computed in its entirety only on one core in
    the XLA device.
    Args:
      core: The core to assign this Op to.
    """
        return Sharding(proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.MAXIMAL, tile_assignment_dimensions=[1], tile_assignment_devices=[core]))

    @classmethod
    def tile(cls, tile_assignment):
        """Returns a Tiled sharding attribute.

    This causes an op to be partially computed on multiple cores in the
    XLA device.

    Args:
      tile_assignment: An np.ndarray describing the topology of the tiling and
        which device will compute which part of the topology.

    Raises:
      TypeError: tile_assignment was not of np.array type.

    TODO(jmolloy): This concept is nefarious and is not
    something we really want to expose to users (especially as the
    contract for tile_assignment is very strict).
    """
        if not isinstance(tile_assignment, _np.ndarray):
            raise TypeError('Tile assignment must be of type np.ndarray')
        dims = list(tile_assignment.shape)
        flattened_devices = tile_assignment.reshape(-1, order='C')
        return Sharding(proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.OTHER, tile_assignment_dimensions=dims, tile_assignment_devices=list(flattened_devices)))

    @classmethod
    def subgroup_tile(cls, tile_assignment, subgroup_modes):
        """Returns a subgroup manual sharding attribute.

    This is similar to tile(), but tile_assignment has one or more dimension
    than the tensor, and subgroup_modes define the sharding types in the last
    dimensions of tile_assignment.

    Args:
      tile_assignment: An np.ndarray describing the topology of the tiling and
        which device will compute which part of the topology.
      subgroup_modes: sharding types for the dimension more than the tensor
        shape rank.

    Raises:
      TypeError: tile_assignment was not of np.array type or subgroup_modes
        has unsupported sharding type.
    """
        if not isinstance(tile_assignment, _np.ndarray):
            raise TypeError('SubgroupTile assignment must be of type np.ndarray')
        if not isinstance(subgroup_modes, list):
            raise TypeError('subgroup_modes in subgroup manual must be of type list')
        if len(tile_assignment.shape) < len(subgroup_modes):
            raise TypeError('SubgroupTile assignment must have rank larger than length of subgroup_modes')
        for sharding_type in subgroup_modes:
            if sharding_type not in [xla_data_pb2.OpSharding.REPLICATED, xla_data_pb2.OpSharding.MANUAL]:
                raise TypeError('Each sharding_type in subgroup_modes in subgroup manual must be of type xla_data_pb2.OpSharding.REPLICATED or xla_data_pb2.OpSharding.MANUAL')
        dims = list(tile_assignment.shape)
        flattened_devices = tile_assignment.reshape(-1, order='C')
        return Sharding(proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.OTHER, tile_assignment_dimensions=dims, tile_assignment_devices=list(flattened_devices), last_tile_dims=list(subgroup_modes)))

    @classmethod
    def partial_tile(cls, tile_assignment):
        """Returns a partially tiled sharding attribute.

    This is similar to tile(), but tile_assignment has one more dimension than
    the tensor, and tiles in the last dimension of tile_assignment are
    replicated.

    Args:
      tile_assignment: An np.ndarray describing the topology of the tiling and
        which device will compute which part of the topology.

    Raises:
      TypeError: tile_assignment was not of np.array type.
    """
        if not isinstance(tile_assignment, _np.ndarray):
            raise TypeError('PartialTile assignment must be of type np.ndarray')
        dims = list(tile_assignment.shape)
        flattened_devices = tile_assignment.reshape(-1, order='C')
        return Sharding(proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.OTHER, tile_assignment_dimensions=dims, tile_assignment_devices=list(flattened_devices), replicate_on_last_tile_dim=True))

    @classmethod
    def split(cls, tensor, split_dimension, num_devices, input_shape=None):
        """Returns a Sharding that splits a tensor across a dimension.

    This creates a Tiled attribute, similar to tile(), but easier to use for the
    common case of tiling a tensor N ways in one dimension.

    Args:
      tensor: A tf.Tensor to split.
      split_dimension: The dimension number to split.
      num_devices: The number of cores to split `tensor` over.
      input_shape: The shape of the original tensor.

    Raises:
      ValueError: The tensor to split was smaller in the split dimension than
        the number of devices to split over.
    """
        if input_shape:
            shape = input_shape
        else:
            shape = tensor.shape.as_list()
        if shape[split_dimension] is not None and shape[split_dimension] < num_devices:
            raise ValueError('Split dimension was smaller than the required number of splits: shape=%r, dimension=%r, num_devices=%r' % (shape, split_dimension, num_devices))
        tile_assignment_dims = [1] * len(shape)
        tile_assignment_dims[split_dimension] = num_devices
        return Sharding(proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.OTHER, tile_assignment_dimensions=tile_assignment_dims, tile_assignment_devices=range(num_devices)))

    def apply_to_tensor(self, tensor, assign_tuple_sharding=False, use_sharding_op=False, unspecified_dims=None):
        """Applies this Sharding attribute to `tensor`.

    Args:
      tensor: A tf.Tensor to split.
      assign_tuple_sharding: If the sharding type should be a tuple.
      use_sharding_op: Whether to create a sharding op on `tensor`.
      unspecified_dims: An optional list of dimensions unspecified.

    Returns:
      The tensor with Sharding attribute.
    """
        if unspecified_dims:
            assert use_sharding_op and (not assign_tuple_sharding)
        proto = self._proto
        if use_sharding_op:
            if assign_tuple_sharding:
                proto = self._create_tuple_proto(num_outputs=1)
                tensor = tf2xla.sharding(tensor, sharding=proto.SerializeToString())
            else:
                tensor = tf2xla.sharding(tensor, sharding=proto.SerializeToString(), unspecified_dims=unspecified_dims or [])
        elif assign_tuple_sharding or len(tensor.op.outputs) > 1:
            proto = self._get_or_create_tuple_proto(tensor.op)
            tuple_shardings = list(proto.tuple_shardings)
            tuple_shardings[tensor.value_index] = self._proto
            proto = xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.TUPLE, tuple_shardings=tuple_shardings)
        tensor.op._set_attr('_XlaSharding', attr_value_pb2.AttrValue(s=proto.SerializeToString()))
        return tensor

    def apply_to_operation(self, operation):
        """Applies this Sharding attribute to `operation`.

    Args:
      operation: A tf.Operation to add sharding annotation.
    """
        attr_value = attr_value_pb2.AttrValue(s=self._proto.SerializeToString())
        operation._set_attr('_XlaSharding', attr_value)

    @property
    def proto(self):
        """Return the sharding protobuf of type xla_data_pb2.OpSharding."""
        return self._proto

    def _get_or_create_tuple_proto(self, op):
        try:
            attr = op.get_attr('_XlaSharding')
            proto = xla_data_pb2.OpSharding()
            proto.ParseFromString(attr)
            return proto
        except ValueError:
            return self._create_tuple_proto(len(op.outputs))

    def _create_tuple_proto(self, num_outputs):
        shardings = [xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED)] * num_outputs
        return xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.TUPLE, tuple_shardings=shardings)