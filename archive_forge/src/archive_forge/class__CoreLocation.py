import functools
import time
from typing import List, Optional, Dict
import numpy as np
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.util.tf_export import tf_export
class _CoreLocation:
    """Represents a TPU core's location in the mesh."""

    def __init__(self, x: int=0, y: int=0, z: int=0, core: int=0):
        self.x = x
        self.y = y
        self.z = z
        self.core = core

    def __eq__(self, other):
        if not isinstance(other, _CoreLocation):
            return False
        return self.x == other.x and self.y == other.y and (self.z == other.z) and (self.core == other.core)

    def __ne__(self, other):
        if not isinstance(other, _CoreLocation):
            return True
        return not self == other

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.core))

    def __repr__(self):
        return f'{type(self).__name__}(x={self.x}, y={self.y}, z={self.z}, core={self.core})'

    def to_list(self):
        return [self.x, self.y, self.z, self.core]