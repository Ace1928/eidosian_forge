import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_anonymous_tensor_operand(self, oper):
    assert isinstance(oper, Operand)
    operand_id = self.get_next_operand_id()
    self.operands.append(oper)
    return operand_id