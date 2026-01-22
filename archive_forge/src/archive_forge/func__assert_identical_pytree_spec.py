from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
def _assert_identical_pytree_spec(spec1: pytree.TreeSpec, spec2: pytree.TreeSpec, error_message: str) -> None:
    """Assert the two `TreeSpec` objects are identical.

    Args:
        spec1: The first `TreeSpec` object.
        spec2: The second `TreeSpec` object.
        error_message: The error message to raise if the two `TreeSpec` objects are not
            identical.

    Raises:
        ValueError: If the two `TreeSpec` objects are not identical.
    """
    pass_if_any_checks: Sequence[Callable[[], bool]] = [lambda: spec1 == spec2, lambda: _replace_tuple_with_list(spec1) == _replace_tuple_with_list(spec2), lambda: _open_top_level_list_if_single_element(spec1) == spec2, lambda: spec1 == _open_top_level_list_if_single_element(spec2)]
    if not any((check() for check in pass_if_any_checks)):
        raise ValueError(f'{error_message}\nExpect {spec1}.\nActual {spec2}.')