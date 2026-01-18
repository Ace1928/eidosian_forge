import torch._C._onnx as _C_onnx
from torch.onnx import _constants
@training_mode.setter
def training_mode(self, training_mode: _C_onnx.TrainingMode):
    if not isinstance(training_mode, _C_onnx.TrainingMode):
        raise TypeError("training_mode must be of type 'torch.onnx.TrainingMode'. This is likely a bug in torch.onnx.")
    self._training_mode = training_mode