from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
The list of replaced weight tensors.