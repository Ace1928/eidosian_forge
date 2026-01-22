from __future__ import annotations
import textwrap
from typing import Optional
from torch import _C
from torch.onnx import _constants
from torch.onnx._internal import diagnostics
class OnnxExporterWarning(UserWarning):
    """Base class for all warnings in the ONNX exporter."""
    pass