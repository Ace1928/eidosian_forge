import logging
from typing import Iterable, Optional, Set, Tuple
from pip._internal.build_env import BuildEnvironment
from pip._internal.distributions.base import AbstractDistribution
from pip._internal.exceptions import InstallationError
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.subprocess import runner_with_spinner_message
def _raise_conflicts(self, conflicting_with: str, conflicting_reqs: Set[Tuple[str, str]]) -> None:
    format_string = 'Some build dependencies for {requirement} conflict with {conflicting_with}: {description}.'
    error_message = format_string.format(requirement=self.req, conflicting_with=conflicting_with, description=', '.join((f'{installed} is incompatible with {wanted}' for installed, wanted in sorted(conflicting_reqs))))
    raise InstallationError(error_message)