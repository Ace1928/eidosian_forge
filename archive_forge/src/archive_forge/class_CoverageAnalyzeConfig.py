from __future__ import annotations
import typing as t
from .. import (
class CoverageAnalyzeConfig(CoverageConfig):
    """Configuration for the `coverage analyze` command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args)
        self.display_stderr = True