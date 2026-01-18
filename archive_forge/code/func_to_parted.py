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
def to_parted(self) -> 'Layout':
    """Returns a "parted" layout from a static layout.

    A parted layout contains axes that are treated as independent by most of
    SPMD expanders.

    FIXME(b/285905569): The exact semantics is still being investigated.
    """
    return Layout._new_object(layout=super().to_parted())