import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
def _shorten_file_name(self, full_file_name: str, truncate_to_last_n: int=2):
    splits = full_file_name.split('/')
    if len(splits) >= truncate_to_last_n:
        return '/'.join(splits[-truncate_to_last_n:])
    return full_file_name