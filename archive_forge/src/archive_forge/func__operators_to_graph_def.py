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
def _operators_to_graph_def(shapes, ops, colon_replacement='$', with_ssa=True, with_gradient_scope=True, blob_name_tracker=None, show_simplified=False, custom_rename=None):
    """
    Convert a set of operators to a graph using the main function.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        ops: List of Caffe2 operators, representing some computation graph
        ### **kwargs (model_to_graph_def, nets_to_graph_def, protos_to_graph_def) ###
        colon_replacement: Symbol to replace ':' with. ':i' in TF has a special
            meaning, so we need to replace it with a non-conflicting symbol.
        with_ssa: Boolean
        with_gradient_scope: Boolean
        blob_name_tracker: Dictionary tracking names of blobs (inputs/outputs
            from operators)
        show_simplified: Whether to show a simplified version of the model graph
            Sets all of the following values:
                clear_debug_info: Boolean representing whether to silence debug
                    info (which can be very verbose)
                show_forward_only: Boolean representing whether to only show
                    blobs involved in the forward pass
                show_cpu_only: Boolean representing whether to only show blobs
                    that are not associated with a gpu
                use_tensorflow_naming: Boolean representing whether to convert
                    some common Caffe2 naming conventions to their Tensorflow
                    counterparts
        custom_rename: Function string -> string that defines a custom
            renaming function to use.

    Returns:
        current_graph: GraphDef representing the computation graph formed by the
            set of operators.
    """
    if blob_name_tracker is not None:
        blob_name_tracker.clear()
    else:
        blob_name_tracker = {}
    blob_name_tracker.update(_get_blob_names(ops))
    _clear_debug_info(ops, show_simplified)
    ops = _filter_ops(ops, _check_if_forward, show_simplified)
    ops = _filter_ops(ops, _check_if_cpu, show_simplified)
    if custom_rename:
        _rename_all(shapes, blob_name_tracker, ops, custom_rename)
    if colon_replacement:
        _replace_colons(shapes, blob_name_tracker, ops, colon_replacement)
    if with_ssa:
        _convert_to_ssa(shapes, blob_name_tracker, ops)
    if with_gradient_scope:
        _add_gradient_scope(shapes, blob_name_tracker, ops)
    _fill_missing_operator_names(ops)
    if show_simplified:
        _rename_tensorflow_style(shapes, blob_name_tracker, ops)
    producing_ops: Dict[caffe2_pb2.OperatorDef, List] = {}
    blobs = set()
    input_blobs, inter_blobs, _ = _compute_in_out(ops)
    current_graph = GraphDef()
    seen = set(input_blobs)
    for op in ops:
        nodes_from_op = _operator_to_node_simp(op, inter_blobs, seen) if show_simplified else [_operator_to_node(shapes, op)]
        current_graph.node.extend(nodes_from_op)
        for input_blob in op.input:
            blobs.add(input_blob)
        for i, output_blob in enumerate(op.output):
            blobs.add(output_blob)
            producing_ops.setdefault(output_blob, []).append((op, i))
    if show_simplified:
        blobs = input_blobs
    for blob in sorted(blobs):
        current_graph.node.extend([_blob_to_node(producing_ops, {}, blob)])
    return current_graph