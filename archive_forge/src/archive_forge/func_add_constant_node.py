import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_constant_node(self, node):
    assert node.inputsSize() == 0
    assert node.outputsSize() == 1
    output = node.outputsAt(0)
    ctype = output.type()
    value = output.toIValue()
    self.add_constant_value(output, ctype, value)