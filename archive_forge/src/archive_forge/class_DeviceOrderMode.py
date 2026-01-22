import enum
import math
from typing import List, Optional, Tuple
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.topology import Topology
from tensorflow.python.util.tf_export import tf_export
@tf_export('tpu.experimental.DeviceOrderMode')
class DeviceOrderMode(enum.IntEnum):
    """The way of determining device orders when computing device assignment."""
    AUTO = 0
    RING = 1
    MESH = 2