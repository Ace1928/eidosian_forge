import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _take_tensors(tensors, size_limit):
    """Group tensors into chunks. This generator yields a chunk at each time,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Yields:
        Blocks of tensors of same type and within size_limit. The yielded
        tensors are only ordered as the original sequence within its types.
    """
    buf_dict: DefaultDict[str, List] = defaultdict(lambda: [[], 0])
    for tensor in tensors:
        t = tensor.type()
        if tensor.is_sparse:
            indices = torch.Tensor._indices(tensor)
            values = torch.Tensor._values(tensor)
            size = indices.numel() * indices.element_size() + values.numel() * values.element_size()
        else:
            size = tensor.numel() * tensor.element_size()
        buf_and_size = buf_dict[t]
        if buf_and_size[1] + size > size_limit and buf_and_size[1] > 0:
            yield buf_and_size[0]
            buf_and_size = buf_dict[t] = [[], 0]
        buf_and_size[0].append(tensor)
        buf_and_size[1] += size
    for buf, _ in buf_dict.values():
        if len(buf) > 0:
            yield buf