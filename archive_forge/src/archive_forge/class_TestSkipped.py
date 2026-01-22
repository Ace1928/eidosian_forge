from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
class TestSkipped(TestResult):
    """Test skipped."""

    def __init__(self, command: str, test: str, python_version: t.Optional[str]=None) -> None:
        super().__init__(command, test, python_version)
        self.reason: t.Optional[str] = None

    def write_console(self) -> None:
        """Write results to console."""
        if self.reason:
            display.warning(self.reason)
        else:
            display.info('No tests applicable.', verbosity=1)

    def write_junit(self, args: TestConfig) -> None:
        """Write results to a junit XML file."""
        test_case = junit_xml.TestCase(classname=self.command, name=self.name, skipped=self.reason or 'No tests applicable.')
        self.save_junit(args, test_case)