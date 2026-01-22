from __future__ import annotations
import typing
import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.model import expect
def SequenceEraseImpl(sequence: list[np.ndarray], position: int | None=None) -> list[np.ndarray | None]:
    if position is None:
        position = -1
    del sequence[position]
    return sequence