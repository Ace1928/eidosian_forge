from __future__ import annotations
import os
from ....config import (
from ....containers import (
from . import (
class ACMEEnvironment(CloudEnvironment):
    """ACME environment plugin. Updates integration test environment after delegation."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""
        ansible_vars = dict(acme_host=self._get_cloud_config('acme_host'))
        return CloudEnvironmentConfig(ansible_vars=ansible_vars)