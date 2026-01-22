from __future__ import annotations
import abc
import glob
import hashlib
import json
import os
import pathlib
import re
import collections
import collections.abc as c
import typing as t
from ...constants import (
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...executor import (
from ...python_requirements import (
from ...config import (
from ...test import (
from ...data import (
from ...content_config import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...venv import (
class SanityMultipleVersion(SanityTest, metaclass=abc.ABCMeta):
    """Base class for sanity test plugins which should run on multiple python versions."""

    @abc.abstractmethod
    def test(self, args: SanityConfig, targets: SanityTargets, python: PythonConfig) -> TestResult:
        """Run the sanity test and return the result."""

    def load_processor(self, args: SanityConfig, python_version: str) -> SanityIgnoreProcessor:
        """Load the ignore processor for this sanity test."""
        return SanityIgnoreProcessor(args, self, python_version)

    @property
    def needs_pypi(self) -> bool:
        """True if the test requires PyPI, otherwise False."""
        return False

    @property
    def supported_python_versions(self) -> t.Optional[tuple[str, ...]]:
        """A tuple of supported Python versions or None if the test does not depend on specific Python versions."""
        return SUPPORTED_PYTHON_VERSIONS

    def filter_targets_by_version(self, args: SanityConfig, targets: list[TestTarget], python_version: str) -> list[TestTarget]:
        """Return the given list of test targets, filtered to include only those relevant for the test, taking into account the Python version."""
        if not python_version:
            raise Exception('python_version is required to filter multi-version tests')
        targets = super().filter_targets_by_version(args, targets, python_version)
        if python_version in REMOTE_ONLY_PYTHON_VERSIONS:
            content_config = get_content_config(args)
            if python_version not in content_config.modules.python_versions:
                return []
            targets = self.filter_remote_targets(targets)
        return targets