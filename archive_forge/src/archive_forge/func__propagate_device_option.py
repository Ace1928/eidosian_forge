import copy
import logging
import os
import re
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from typing import Set, Dict, Tuple, List
def _propagate_device_option(net_def):
    """
    Propagate the device options from net to operators.

    Args:
        net_def: A caffe2_pb2.NetDef representing a computation graph. The graph
            consists of Caffe2 operators.

    Returns:
        None. Iterates through all ops contained within the net. For each op,
            modifies the op device_option in-place to be the net device_option
            if the op has no pre-existing device_option, and leaves the op as-is
            if it already has a device_option.
    """
    if not net_def.HasField('device_option'):
        return
    for op in net_def.op:
        if not op.HasField('device_option'):
            op.device_option.CopyFrom(net_def.device_option)