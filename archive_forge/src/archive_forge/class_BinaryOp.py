import dis
import enum
import opcode as _opcode
import sys
from abc import abstractmethod
from dataclasses import dataclass
from marshal import dumps as _dumps
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union
import bytecode as _bytecode
@enum.unique
class BinaryOp(enum.IntEnum):
    ADD = 0
    AND = 1
    FLOOR_DIVIDE = 2
    LSHIFT = 3
    MATRIX_MULTIPLY = 4
    MULTIPLY = 5
    REMAINDER = 6
    OR = 7
    POWER = 8
    RSHIFT = 9
    SUBTRACT = 10
    TRUE_DIVIDE = 11
    XOR = 12
    INPLACE_ADD = 13
    INPLACE_AND = 14
    INPLACE_FLOOR_DIVIDE = 15
    INPLACE_LSHIFT = 16
    INPLACE_MATRIX_MULTIPLY = 17
    INPLACE_MULTIPLY = 18
    INPLACE_REMAINDER = 19
    INPLACE_OR = 20
    INPLACE_POWER = 21
    INPLACE_RSHIFT = 22
    INPLACE_SUBTRACT = 23
    INPLACE_TRUE_DIVIDE = 24
    INPLACE_XOR = 25