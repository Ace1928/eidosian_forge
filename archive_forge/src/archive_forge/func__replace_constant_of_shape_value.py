from typing import List, Optional, Union
import numpy as np
from onnx import (
from onnx.helper import (
from onnx.numpy_helper import from_array
def _replace_constant_of_shape_value(onx: Union[GraphProto, FunctionProto], value_constant_of_shape: float) -> Union[GraphProto, FunctionProto]:
    """Replaces all fill value of all nodes *ConstantOfShape*."""
    if isinstance(onx, GraphProto):
        nodes = list(onx.node)
    elif isinstance(onx, FunctionProto):
        nodes = list(onx.node)
    else:
        raise TypeError(f'Not implemented for type {type(onx)}.')
    existing_names = set()
    for node in nodes:
        existing_names |= set(node.input)
        existing_names |= set(node.output)
    update = {}
    for inode, node in enumerate(nodes):
        if node.op_type != 'ConstantOfShape':
            continue
        tensor = node.attribute[0].t
        new_tensor = make_tensor(tensor.name, tensor.data_type, [1], [value_constant_of_shape])
        new_node = make_node('ConstantOfShape', node.input, node.output)
        att = make_attribute(node.attribute[0].name, value=new_tensor)
        new_node.attribute.append(att)
        update[inode] = new_node
    for inode, up in update.items():
        nodes[inode] = up
    if isinstance(onx, GraphProto):
        graph = make_graph(nodes, onx.name, onx.input, onx.output, initializer=onx.initializer, sparse_initializer=onx.sparse_initializer)
        return graph
    if isinstance(onx, FunctionProto):
        new_onx = make_function(onx.domain, onx.name, onx.input, onx.output, nodes, opset_imports=onx.opset_import)
        return new_onx
    raise TypeError(f'Not implemented for type {type(onx)}.')