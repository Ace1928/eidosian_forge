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
def _check_if_cpu(blob):
    """
    Check if the blob's name starts with '_gpu'.

    Args:
        blob: The blob to inspect

    Returns:
        Boolean representing whether this blob is associated with a gpu
    """
    return not blob.startswith('_gpu')