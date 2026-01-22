from __future__ import annotations
import collections.abc as c
import json
import os
import re
import typing as t
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...config import (
from ...python_requirements import (
from ...target import (
from ...data import (
from ...pypi_proxy import (
from ...provisioning import (
from ...coverage_util import (
class PathChecker:
    """Checks code coverage paths to verify they are valid and reports on the findings."""

    def __init__(self, args: CoverageConfig, collection_search_re: t.Optional[t.Pattern]=None) -> None:
        self.args = args
        self.collection_search_re = collection_search_re
        self.invalid_paths: list[str] = []
        self.invalid_path_chars = 0

    def check_path(self, path: str) -> bool:
        """Return True if the given coverage path is valid, otherwise display a warning and return False."""
        if os.path.isfile(to_bytes(path)):
            return True
        if self.collection_search_re and self.collection_search_re.search(path) and (os.path.basename(path) == '__init__.py'):
            return False
        self.invalid_paths.append(path)
        self.invalid_path_chars += len(path)
        if self.args.verbosity > 1:
            display.warning('Invalid coverage path: %s' % path)
        return False

    def report(self) -> None:
        """Display a warning regarding invalid paths if any were found."""
        if self.invalid_paths:
            display.warning('Ignored %d characters from %d invalid coverage path(s).' % (self.invalid_path_chars, len(self.invalid_paths)))