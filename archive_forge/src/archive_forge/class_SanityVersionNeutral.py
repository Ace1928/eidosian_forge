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
class SanityVersionNeutral(SanityTest, metaclass=abc.ABCMeta):
    """Base class for sanity test plugins which are idependent of the python version being used."""

    @abc.abstractmethod
    def test(self, args: SanityConfig, targets: SanityTargets) -> TestResult:
        """Run the sanity test and return the result."""

    def load_processor(self, args: SanityConfig) -> SanityIgnoreProcessor:
        """Load the ignore processor for this sanity test."""
        return SanityIgnoreProcessor(args, self, None)

    @property
    def supported_python_versions(self) -> t.Optional[tuple[str, ...]]:
        """A tuple of supported Python versions or None if the test does not depend on specific Python versions."""
        return None