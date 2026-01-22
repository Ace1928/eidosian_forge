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
def _add_gradient_scope(shapes, blob_name_tracker, ops):
    """
    For all operators or blobs with name containing "_grad", add a "GRADIENTS/" scope.

    Note: breaks graph execution since the blob -> gradient mapping is
    hardcoded.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        blob_name_tracker: Dictionary of all unique blob names (with respect to
            some context).
        ops: List of Caffe2 operators

    Returns:
        None. Modifies shapes, blob_name_tracker and ops in-place by renaming.
    """

    def f(name):
        if '_grad' in name:
            return f'GRADIENTS/{name}'
        else:
            return name
    _rename_all(shapes, blob_name_tracker, ops, f)