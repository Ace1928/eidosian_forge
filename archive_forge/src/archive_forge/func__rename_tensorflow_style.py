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
def _rename_tensorflow_style(shapes, blob_name_tracker, ops):
    """
    Convert some of the common names in Caffe2 to tensorflow.

    NOTE: The common names in both Caffe2 and Tensorflow are currently
        hardcoded, if either side changes at some point, then this code should
        change as well.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        blob_name_tracker: Dictionary of all unique blob names (with respect to
            some context).
        ops: List of Caffe2 operators

    Returns:
        None. The _rename_all() call modifies blob_name_tracker and ops in-place.
    """
    WEIGHT = re.compile('(_w)$')
    WEIGHT_ = re.compile('(_w_)')
    BN = re.compile('(_bn)$')
    BN_ = re.compile('(_bn_)')
    BIAS = re.compile('(_b)$')
    BIAS_ = re.compile('(_b_)')
    SCALE = re.compile('(_s)$')
    SCALE_ = re.compile('(_s_)')
    SUM = re.compile('(_sum)$')
    SUM_ = re.compile('(_sum_)')
    BRANCH = re.compile('(_branch)')

    def f(name):
        inter_name = WEIGHT_.sub('/weight_', WEIGHT.sub('/weight', name))
        inter_name = BN_.sub('/batchnorm_', BN.sub('/batchnorm', inter_name))
        inter_name = BIAS_.sub('/bias_', BIAS.sub('/bias', inter_name))
        inter_name = SCALE_.sub('/scale_', SCALE.sub('/scale', inter_name))
        inter_name = SUM_.sub('/sum_', SUM.sub('/sum', inter_name))
        new_name = BRANCH.sub('/branch', inter_name)
        return new_name
    _rename_all(shapes, blob_name_tracker, ops, f)