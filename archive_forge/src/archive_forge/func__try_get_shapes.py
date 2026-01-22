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
def _try_get_shapes(nets):
    """
    Get missing shapes for all blobs contained in the nets.

    Args:
        nets: List of core.Net to extract blob shape information from.

    Returns:
        Dictionary containing blob name to shape/dimensions mapping. The net
            is a computation graph that is composed of operators, and the
            operators have input and output blobs, each with their own dims.
    """
    try:
        shapes, _ = workspace.InferShapesAndTypes(nets)
        return shapes
    except Exception as e:
        log.warning('Failed to compute shapes: %s', e)
        return {}