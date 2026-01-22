import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_list_construct(self, node):
    assert node.outputsSize() == 1
    output = node.outputsAt(0)
    ctype = output.type()
    const_vals: Optional[List] = []
    tensors: Optional[List] = []
    for inp in node.inputs():
        if const_vals is not None and inp in self.constants:
            _, val = self.get_constant_value(inp)
            const_vals.append(val)
        else:
            const_vals = None
        if tensors is not None and inp.type().kind() == 'TensorType':
            tensors.append(inp)
        else:
            tensors = None
    if const_vals is not None:
        self.add_constant_value(output, ctype, const_vals)
    if tensors is not None:
        self.add_tensor_sequence(output, tensors)
    if const_vals is None and tensors is None:
        raise Exception(f'Unable to handle ListConstruct node.  Neither all constants nor all tensors. {node!r}')