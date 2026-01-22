import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
def _parse_topology(self, serialized):
    """Parses a serialized `TopologyProto` into `self`."""
    proto = topology_pb2.TopologyProto()
    proto.ParseFromString(serialized)
    self._mesh_shape = np.array(proto.mesh_shape, dtype=np.int32)
    if len(self._mesh_shape) != 4 or any(self._mesh_shape < 1):
        raise ValueError('`mesh_shape` must be a vector of size 4 with positive entries; got {}'.format(self._mesh_shape))
    if proto.num_tasks < 0:
        raise ValueError('`num_tasks` must be >= 0; got {}'.format(proto.num_tasks))
    if proto.num_tpu_devices_per_task < 0:
        raise ValueError('`num_tpu_devices_per_task` must be >= 0; got {}'.format(proto.num_tpu_devices_per_task))
    expected_coordinates_size = proto.num_tasks * proto.num_tpu_devices_per_task * len(proto.mesh_shape)
    if len(proto.device_coordinates) != expected_coordinates_size:
        raise ValueError('`device_coordinates` must have shape num_tasks ({}) * num_tpu_devices_per_task ({}) * len(mesh_shape) ({}); got shape {}'.format(proto.num_tasks, proto.num_tpu_devices_per_task, proto.mesh_shape, len(proto.device_coordinates)))
    coords = np.array(proto.device_coordinates, dtype=np.int32)
    if any(coords < 0):
        raise ValueError('All values in `device_coordinates` must be >= 0, got {}'.format(coords))
    coords = coords.reshape((proto.num_tasks, proto.num_tpu_devices_per_task, len(proto.mesh_shape)))
    self._device_coordinates = coords