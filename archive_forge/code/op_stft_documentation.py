import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_concat_from_sequence import _concat_from_sequence
from onnx.reference.ops.op_dft import _cfft as _dft
from onnx.reference.ops.op_slice import _slice
Reverses of `stft`.