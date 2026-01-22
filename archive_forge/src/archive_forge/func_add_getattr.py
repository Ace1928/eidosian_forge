import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_getattr(self, node):
    assert node.inputsSize() == 1
    assert node.outputsSize() == 1
    obj_ctype, obj = self.get_constant_value(node.inputsAt(0))
    assert str(obj_ctype).startswith('__torch__.')
    name = node.s('name')
    value = getattr(obj, name)
    output = node.outputsAt(0)
    ctype = output.type()
    self.add_constant_value(output, ctype, value)