import collections
import functools
import itertools
from typing import List, Dict, Optional, Union
import numpy as np
from tensorflow.dtensor.proto import layout_pb2
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util.tf_export import tf_export
def unravel_index(self):
    """Returns a dictionary from device ID to {dim_name: dim_index}.

    For example, for a 3x2 mesh, return this:

    ```
      { 0: {'x': 0, 'y', 0},
        1: {'x': 0, 'y', 1},
        2: {'x': 1, 'y', 0},
        3: {'x': 1, 'y', 1},
        4: {'x': 2, 'y', 0},
        5: {'x': 2, 'y', 1} }
    ```
    """
    idx_ranges = [range(self.dim_size(dim_name)) for dim_name in self.dim_names]
    mesh_pos = itertools.product(*idx_ranges)
    mapping = {}
    for device_id, device_pos in enumerate(mesh_pos):
        device_loc = {}
        for dim_name, dim_index in zip(self.dim_names, device_pos):
            device_loc[dim_name] = dim_index
        mapping[device_id] = device_loc
    return mapping