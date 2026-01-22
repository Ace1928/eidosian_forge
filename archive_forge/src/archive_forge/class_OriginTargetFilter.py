from __future__ import annotations
import abc
import typing as t
from ...config import (
from ...util import (
from ...target import (
from ...host_configs import (
from ...host_profiles import (
class OriginTargetFilter(PosixTargetFilter[OriginConfig]):
    """Target filter for localhost."""

    def filter_targets(self, targets: list[IntegrationTarget], exclude: set[str]) -> None:
        """Filter the list of targets, adding any which this host profile cannot support to the provided exclude list."""
        super().filter_targets(targets, exclude)
        arch = detect_architecture(self.config.python.path)
        if arch:
            self.skip(f'skip/{arch}', f'which are not supported by {arch}', targets, exclude)