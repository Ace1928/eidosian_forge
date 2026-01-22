import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple
import numpy as np
import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper
from ..utils import logging, recurse_getattr
def _deduplicate_gather_matmul(model: ModelProto, tied_groups_to_tie: List[List[str]], tied_groups_map: Dict[Tuple[str], List[Dict[str, Any]]], initializer_name_to_idx: Dict[str, int]):
    """
    Removes the duplicate initializers for Gather and MatMul from the ONNX model based on the information in tied_groups_map i.e. of which ONNX initializers correspond to a single torch parameter.
    """
    node_name_to_idx = {}
    for idx, node in enumerate(model.graph.node):
        node_name_to_idx[node.name] = idx
    for params in tied_groups_to_tie:
        torch_to_initializer = tied_groups_map[tuple(params)]
        ref_idx = None
        for i in range(len(torch_to_initializer)):
            ops_using_initializer = set()
            for node_name in torch_to_initializer[i]['nodes_containing_initializer']:
                ops_using_initializer.add(model.graph.node[node_name_to_idx[node_name]].op_type)
            if ops_using_initializer == {'Gather'}:
                ref_idx = i
                break
        if ref_idx is None:
            logger.warning(f'Could not deduplicate initializers corresponding to the torch tied parameters {params} as an initializer used only by Gather nodes could not be found. Skipping deduplication.')
            continue
        ref_initializer_name = next(iter(torch_to_initializer[ref_idx]['initializer_name']))
        ref_initializer_idx = initializer_name_to_idx[ref_initializer_name]
        ref_initializer = model.graph.initializer[ref_initializer_idx]
        ref_type = ref_initializer.data_type
        ref_data = numpy_helper.to_array(ref_initializer)
        for i in range(len(torch_to_initializer)):
            if i == ref_idx:
                continue
            initializer_name = next(iter(torch_to_initializer[i]['initializer_name']))
            initializer_idx = initializer_name_to_idx[initializer_name]
            initializer = model.graph.initializer[initializer_idx]
            initializer_type = initializer.data_type
            initializer_data = numpy_helper.to_array(initializer)
            if initializer_name == ref_initializer_name:
                continue
            if ref_type == initializer_type and np.array_equal(ref_data, initializer_data):
                logger.info(f'Removing duplicate initializer {initializer_name}...')
                for node in model.graph.node:
                    if initializer_name in node.input:
                        input_idx = list(node.input).index(initializer_name)
                        node.input[input_idx] = ref_initializer_name
                model.graph.initializer.pop(initializer_idx)
            elif ref_type == initializer_type and np.array_equal(ref_data.T, initializer_data):
                logger.info(f'Removing duplicate initializer {initializer_name}...')
                transpose_output_name = f'{ref_initializer_name}_transposed'
                transpose_node_name = f'Transpose_{len(model.graph.node) + 1}'
                minimum_node_idx = len(model.graph.node)
                for node_idx, node in enumerate(model.graph.node):
                    if initializer_name in node.input:
                        minimum_node_idx = node_idx
                        break
                transpose_node = onnx.helper.make_node('Transpose', name=transpose_node_name, inputs=[ref_initializer_name], outputs=[transpose_output_name])
                model.graph.node.insert(minimum_node_idx, transpose_node)
                for node in model.graph.node:
                    if initializer_name in node.input:
                        input_idx = list(node.input).index(initializer_name)
                        node.input[input_idx] = transpose_output_name
                model.graph.initializer.pop(initializer_idx)
            else:
                logger.warning(f'No deduplication implementation for {initializer_name} although it should be deduplicated. Please open an issue in Optimum repository.')
    return model