from __future__ import annotations
import dataclasses
from typing import Dict
from torch.onnx._internal.fx import _pass, diagnostics, registration
@dataclasses.dataclass
class UnsupportedFxNodesAnalysisResult(_pass.AnalysisResult):
    unsupported_op_to_target_mapping: Dict[str, Dict[str, None]]