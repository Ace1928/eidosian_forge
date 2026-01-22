import textwrap
from collections import deque
from typing import List, Sequence, Type, TypeVar
import torch
from . import (
from .common import AttentionBwOpBase, AttentionFwOpBase, Inputs
def _run_priority_list(name: str, priority_list: Sequence[T], inp: Inputs) -> T:
    not_supported_reasons: List[List[str]] = []
    for op in priority_list:
        not_supported = op.not_supported_reasons(inp)
        if not not_supported:
            return op
        not_supported_reasons.append(not_supported)
    msg = f'No operator found for `{name}` with inputs:\n{textwrap.indent(_format_inputs_description(inp), '     ')}'
    for op, not_supported in zip(priority_list, not_supported_reasons):
        msg += '\n' + _format_not_supported_reasons(op, not_supported)
    raise NotImplementedError(msg)