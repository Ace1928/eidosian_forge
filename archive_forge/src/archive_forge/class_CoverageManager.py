from __future__ import annotations
import abc
import os
import shutil
import tempfile
import typing as t
import zipfile
from ...io import (
from ...ansible_util import (
from ...config import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...host_configs import (
from ...data import (
from ...host_profiles import (
from ...provisioning import (
from ...connections import (
from ...inventory import (
class CoverageManager:
    """Manager for code coverage configuration and state."""

    def __init__(self, args: IntegrationConfig, host_state: HostState, inventory_path: str) -> None:
        self.args = args
        self.host_state = host_state
        self.inventory_path = inventory_path
        if self.args.coverage:
            handler_types = set((get_handler_type(type(profile.config)) for profile in host_state.profiles))
            handler_types.discard(None)
        else:
            handler_types = set()
        handlers = [handler_type(args=args, host_state=host_state, inventory_path=inventory_path) for handler_type in handler_types]
        self.handlers = [handler for handler in handlers if handler.is_active]

    def setup(self) -> None:
        """Perform setup for code coverage."""
        if not self.args.coverage:
            return
        for handler in self.handlers:
            handler.setup()

    def teardown(self) -> None:
        """Perform teardown for code coverage."""
        if not self.args.coverage:
            return
        for handler in self.handlers:
            handler.teardown()

    def get_environment(self, target_name: str, aliases: tuple[str, ...]) -> dict[str, str]:
        """Return a dictionary of environment variables for running tests with code coverage."""
        if not self.args.coverage or 'non_local/' in aliases:
            return {}
        env = {}
        for handler in self.handlers:
            env.update(handler.get_environment(target_name, aliases))
        return env