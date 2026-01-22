import textwrap
from collections import deque
from typing import List, Sequence, Type, TypeVar
import torch
from . import (
from .common import AttentionBwOpBase, AttentionFwOpBase, Inputs
def _is_cutlassB_faster_than_flash(inp: Inputs) -> bool:
    return False