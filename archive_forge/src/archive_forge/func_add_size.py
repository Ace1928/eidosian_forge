import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_size(self, node):
    assert node.inputsSize() == 2
    assert node.outputsSize() == 1
    _, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
    _, value = self.constants[node.inputsAt(1)]
    res = in_oper.shape[value]
    output = node.outputsAt(0)
    self.add_constant_value(output, output.type(), res)