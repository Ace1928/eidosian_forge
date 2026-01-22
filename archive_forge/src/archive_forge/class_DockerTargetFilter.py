from __future__ import annotations
import abc
import typing as t
from ...config import (
from ...util import (
from ...target import (
from ...host_configs import (
from ...host_profiles import (
class DockerTargetFilter(PosixTargetFilter[DockerConfig]):
    """Target filter for docker hosts."""

    def filter_targets(self, targets: list[IntegrationTarget], exclude: set[str]) -> None:
        """Filter the list of targets, adding any which this host profile cannot support to the provided exclude list."""
        super().filter_targets(targets, exclude)
        self.skip('skip/docker', 'which cannot run under docker', targets, exclude)
        if not self.config.privileged:
            self.skip('needs/privileged', 'which require --docker-privileged to run under docker', targets, exclude)