import jax
import numpy as np
from keras.src.utils import jax_utils
def _to_jax_layout(tensor_layout):
    """Convert the TensorLayout to JAX backend specific Sharding.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        A `jax.sharding.NamedSharding` instance.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError('Cannot create sharding when device mesh is not set for TensorLayout.')
    partition_spec = jax.sharding.PartitionSpec(*tensor_layout.axes)
    jax_mesh = _to_jax_mesh(tensor_layout.device_mesh)
    return jax.sharding.NamedSharding(jax_mesh, partition_spec)