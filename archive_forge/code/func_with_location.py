from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
def with_location(self: _Diagnostic, location: infra.Location) -> _Diagnostic:
    """Adds a location to the diagnostic."""
    self.locations.append(location)
    return self