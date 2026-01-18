import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
def tpu_device_name_at_coordinates(self, device_coordinates, job=None):
    """Returns the name of the TPU device assigned to a logical core."""
    return _tpu_device_name(job, self._topology_tasks[tuple(device_coordinates)], self._topology_devices[tuple(device_coordinates)])