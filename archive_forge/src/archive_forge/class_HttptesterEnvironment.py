from __future__ import annotations
import os
from ....util import (
from ....config import (
from ....containers import (
from . import (
class HttptesterEnvironment(CloudEnvironment):
    """HTTP Tester environment plugin. Updates integration test environment after delegation."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""
        return CloudEnvironmentConfig(env_vars=dict(HTTPTESTER='1', KRB5_PASSWORD=str(self._get_cloud_config(KRB5_PASSWORD_ENV))))