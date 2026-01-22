import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def _get_instruction_front(instructions: List[Instruction], idx: int):
    """
    i.e. get the first EXTENDED_ARG instruction (if any) when targeting
    instructions[idx] with a jump.
    """
    target = instructions[idx]
    for offset in (1, 2, 3):
        if idx >= offset and instructions[idx - offset].opcode == dis.EXTENDED_ARG:
            target = instructions[idx - offset]
        else:
            break
    return target