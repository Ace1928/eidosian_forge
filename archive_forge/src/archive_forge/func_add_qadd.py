import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_qadd(self, node, opcode, fuse_code):
    assert node.inputsSize() == 4
    _, scale = self.get_constant_value(node.inputsAt(2), 'FloatType')
    _, zero_point = self.get_constant_value(node.inputsAt(3), 'IntType')
    self._do_add_binary(node, opcode, fuse_code, qparams=(scale, zero_point))