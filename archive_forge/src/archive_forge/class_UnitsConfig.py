from __future__ import annotations
import dataclasses
import enum
import os
import sys
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .data import (
from .host_configs import (
class UnitsConfig(TestConfig):
    """Configuration for the units command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args, 'units')
        self.collect_only: bool = args.collect_only
        self.num_workers: int = args.num_workers
        self.requirements_mode: str = getattr(args, 'requirements_mode', '')
        if self.requirements_mode == 'only':
            self.requirements = True
        elif self.requirements_mode == 'skip':
            self.requirements = False