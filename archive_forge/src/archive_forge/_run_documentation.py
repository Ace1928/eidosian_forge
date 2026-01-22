from __future__ import annotations
import dataclasses
from typing import Any, List, Literal, Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
Describes a single run of an analysis tool, and contains the reported output of that run.