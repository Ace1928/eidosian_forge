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
class CoverageHandler(t.Generic[THostConfig], metaclass=abc.ABCMeta):
    """Base class for configuring hosts for integration test code coverage."""

    def __init__(self, args: IntegrationConfig, host_state: HostState, inventory_path: str) -> None:
        self.args = args
        self.host_state = host_state
        self.inventory_path = inventory_path
        self.profiles = self.get_profiles()

    def get_profiles(self) -> list[HostProfile]:
        """Return a list of profiles relevant for this handler."""
        profile_type = get_generic_type(type(self), HostConfig)
        profiles = [profile for profile in self.host_state.target_profiles if isinstance(profile.config, profile_type)]
        return profiles

    @property
    @abc.abstractmethod
    def is_active(self) -> bool:
        """True if the handler should be used, otherwise False."""

    @abc.abstractmethod
    def setup(self) -> None:
        """Perform setup for code coverage."""

    @abc.abstractmethod
    def teardown(self) -> None:
        """Perform teardown for code coverage."""

    @abc.abstractmethod
    def create_inventory(self) -> None:
        """Create inventory, if needed."""

    @abc.abstractmethod
    def get_environment(self, target_name: str, aliases: tuple[str, ...]) -> dict[str, str]:
        """Return a dictionary of environment variables for running tests with code coverage."""

    def run_playbook(self, playbook: str, variables: dict[str, str]) -> None:
        """Run the specified playbook using the current inventory."""
        self.create_inventory()
        run_playbook(self.args, self.inventory_path, playbook, capture=False, variables=variables)