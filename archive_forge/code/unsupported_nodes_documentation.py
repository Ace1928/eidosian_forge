from __future__ import annotations
import dataclasses
from typing import Dict
from torch.onnx._internal.fx import _pass, diagnostics, registration
Analyze the graph, emit diagnostics and return a result that contains unsupported FX nodes.

        Args:
            diagnostic_level: The diagnostic level to use when emitting diagnostics.

        Returns:
            An analysis result that contains unsupported FX nodes.

        Raises:
            RuntimeErrorWithDiagnostic: If diagnostics are emitted and the diagnostic
                level is `ERROR`.
        