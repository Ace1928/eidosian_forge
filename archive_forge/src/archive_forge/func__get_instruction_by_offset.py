import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def _get_instruction_by_offset(offset_to_inst: Dict[int, Instruction], offset: int):
    """
    Get the instruction located at a given offset, accounting for EXTENDED_ARGs
    """
    for n in (0, 2, 4, 6):
        if offset_to_inst[offset + n].opcode != dis.EXTENDED_ARG:
            return offset_to_inst[offset + n]
    return None