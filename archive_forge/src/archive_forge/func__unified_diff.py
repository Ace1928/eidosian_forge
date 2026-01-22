from __future__ import annotations
import abc
import contextlib
import dataclasses
import difflib
import io
import logging
import sys
from typing import Any, Callable, Optional, Tuple
import torch
import torch.fx
from torch._subclasses import fake_tensor
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import diagnostics, onnxfunction_dispatcher
def _unified_diff(a: str, b: str) -> str:
    """Return a string containing the unified diff of two strings.

    This function calls a patched version of `difflib.unified_diff` with `autojunk` set
    to `False` for `difflib.SequenceMatcher` class. More details can be found in
    `_patch_difflib_sequence_matcher_init` function.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        The unified diff of the two strings. If there is no diff, return "<no diff>".

    Example::

        >>> a = '''class GraphModule(torch.nn.Module):
        ...     def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view = input_ids.view(-1, 3);  input_ids = None
        ... '''
        >>> b = '''class <lambda>(torch.nn.Module):
        ...     def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
        ... '''
        >>> print(_unified_diff(a, b))
        ---
        +++
        @@ -1,4 +1,4 @@
        -class GraphModule(torch.nn.Module):
        -    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        +class <lambda>(torch.nn.Module):
        +    def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
                # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        -        view = input_ids.view(-1, 3);  input_ids = None
        +        view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
    """
    a_list = a.splitlines(keepends=True)
    b_list = b.splitlines(keepends=True)
    with _patch_difflib_sequence_matcher_init():
        diff = ''.join(difflib.unified_diff(a_list, b_list, n=sys.maxsize))
    if not diff:
        return '<no diff>'
    return diff