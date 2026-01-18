import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def transform_code_object(code, transformations, safe=False) -> types.CodeType:
    keys = get_code_keys()
    code_options = {k: getattr(code, k) for k in keys}
    assert len(code_options['co_varnames']) == code_options['co_nlocals']
    instructions = cleaned_instructions(code, safe)
    propagate_line_nums(instructions)
    transformations(instructions, code_options)
    return clean_and_assemble_instructions(instructions, keys, code_options)[1]