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
def _filter_ops(ops, filter_fn, perform_filter):
    """
    Filter unwanted operators based on criteria in 'filter_fn'.

    Args:
        ops: List of Caffe2 operators to filter
        filter_fn: Criteria function for whether inputs/outputs in an operator
            should be filtered.
        perform_filter: Boolean passed from _operators_to_graph_def specifying
            whether to filter operators

    Returns:
        new_ops: Subset of ops containing a subset of their inputs and outputs.
    """
    if not perform_filter:
        return ops
    new_ops = []
    for op in ops:
        inputs = list(op.input)
        outputs = list(op.output)
        del op.input[:]
        del op.output[:]
        new_inputs = [i for i in inputs if filter_fn(i)]
        new_outputs = [o for o in outputs if filter_fn(o)]
        if new_outputs:
            op.input.extend(new_inputs)
            op.output.extend(new_outputs)
            new_ops.append(op)
    return new_ops