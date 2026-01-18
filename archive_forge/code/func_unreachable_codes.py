import copy
import dataclasses
import sys
import types
from typing import Any, cast, Dict, List, Optional, Tuple
from .bytecode_transformation import (
from .utils import ExactWeakKeyDictionary
@staticmethod
def unreachable_codes(code_options) -> List[Instruction]:
    """Codegen a `raise None` to make analysis work for unreachable code"""
    return [create_instruction('LOAD_CONST', argval=None), create_instruction('RAISE_VARARGS', arg=1)]