from __future__ import annotations
import collections.abc as c
import os
import json
import typing as t
from ...target import (
from ...io import (
from ...util import (
from ...util_common import (
from ...executor import (
from ...data import (
from ...host_configs import (
from ...provisioning import (
from . import (
class CoverageCombineConfig(CoverageConfig):
    """Configuration for the coverage combine command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args)
        self.group_by: frozenset[str] = frozenset(args.group_by) if args.group_by else frozenset()
        self.all: bool = args.all
        self.stub: bool = args.stub
        self.export: str = args.export if 'export' in args else False