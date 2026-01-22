from __future__ import annotations
import numpy as np
from onnx.reference.op_run import OpRun
class DFT_17(OpRun):

    def _run(self, x: np.ndarray, dft_length: int | None=None, axis: int=1, inverse: bool=False, onesided: bool=False) -> tuple[np.ndarray]:
        axis = axis % len(x.shape)
        if dft_length is None:
            dft_length = x.shape[axis]
        if inverse:
            result = _cifft(x, dft_length, axis=axis, onesided=onesided)
        else:
            result = _cfft(x, dft_length, axis=axis, onesided=onesided, normalize=False)
        return (result.astype(x.dtype),)