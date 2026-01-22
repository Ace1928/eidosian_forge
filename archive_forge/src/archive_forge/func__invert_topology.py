import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
def _invert_topology(self):
    """Inverts a [task,device,axis] topology to [x,y,z] -> task/device maps."""
    tasks = np.full(list(self.mesh_shape), -1, dtype=np.int32)
    devices = np.full(list(self.mesh_shape), -1, dtype=np.int32)
    for task in range(self.device_coordinates.shape[0]):
        for device in range(self.device_coordinates.shape[1]):
            x, y, z, core = self.device_coordinates[task, device, :]
            tasks[x, y, z, core] = task
            devices[x, y, z, core] = device
    return (tasks, devices)