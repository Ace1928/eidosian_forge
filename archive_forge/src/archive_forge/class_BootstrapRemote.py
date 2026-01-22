from __future__ import annotations
import dataclasses
import os
import typing as t
from .io import (
from .util import (
from .util_common import (
from .core_ci import (
@dataclasses.dataclass
class BootstrapRemote(Bootstrap):
    """Bootstrap remote instances."""
    platform: str
    platform_version: str

    def get_variables(self) -> dict[str, t.Union[str, list[str]]]:
        """The variables to template in the bootstrapping script."""
        variables = super().get_variables()
        variables.update(platform=self.platform, platform_version=self.platform_version)
        return variables