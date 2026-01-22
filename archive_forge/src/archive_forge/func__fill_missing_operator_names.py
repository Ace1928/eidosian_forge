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
def _fill_missing_operator_names(ops):
    """
    Give missing operators a name.

    We expect C2 operators to be generally unnamed. This gives them a scope
    (inferred from their outputs) and a name after their type. Duplicates will
    be postfixed by an index.

    Args:
        ops: List of Caffe2 operators to assign names to.

    Returns:
        None: Modifies 'ops' in-place.
    """
    seen = set()
    for op in ops:
        seen.update(op.input)
        seen.update(op.output)
    for op in ops:
        if op.name:
            name = op.name
        elif op.output or op.input:
            name_list = [os.path.dirname(name) for name in op.output or op.input]
            scope = os.path.commonprefix(name_list)
            name = os.path.join(scope, op.type)
        else:
            name = op.type
        assert name
        op.name = _make_unique_name(seen, name)