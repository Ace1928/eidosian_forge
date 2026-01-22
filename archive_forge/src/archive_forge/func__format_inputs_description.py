import textwrap
from collections import deque
from typing import List, Sequence, Type, TypeVar
import torch
from . import (
from .common import AttentionBwOpBase, AttentionFwOpBase, Inputs
def _format_inputs_description(inp: Inputs) -> str:
    return f'query       : shape={tuple(inp.query.shape)} ({inp.query.dtype})\nkey         : shape={tuple(inp.key.shape)} ({inp.key.dtype})\nvalue       : shape={tuple(inp.value.shape)} ({inp.value.dtype})\nattn_bias   : {type(inp.attn_bias)}\np           : {inp.p}'