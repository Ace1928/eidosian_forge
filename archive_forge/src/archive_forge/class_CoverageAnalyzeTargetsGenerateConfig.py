from __future__ import annotations
import os
import typing as t
from .....encoding import (
from .....data import (
from .....util_common import (
from .....executor import (
from .....provisioning import (
from ... import (
from . import (
from . import (
class CoverageAnalyzeTargetsGenerateConfig(CoverageAnalyzeTargetsConfig):
    """Configuration for the `coverage analyze targets generate` command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args)
        self.input_dir: str = args.input_dir or ResultType.COVERAGE.path
        self.output_file: str = args.output_file