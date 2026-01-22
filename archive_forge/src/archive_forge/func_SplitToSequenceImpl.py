from __future__ import annotations
import typing
import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.model import expect
def SplitToSequenceImpl(tensor: np.ndarray, split: int | list[int] | None=None, axis: int=0, keepdims: int=1) -> list[np.ndarray]:
    dim_size = tensor.shape[axis]
    if split is None:
        split = 1
        split_indices = [i * split + 1 for i in range(dim_size) if i * split + 1 < dim_size]
        if not keepdims:
            results = np.array_split(tensor, split_indices, axis)
            return [np.squeeze(res, axis) for res in results]
    if np.isscalar(split):
        split_indices = [i * split + 1 for i in range(dim_size) if i * split + 1 < dim_size]
    else:
        split_indices = np.cumsum(split) + 1
    return np.array_split(tensor, split_indices, axis)