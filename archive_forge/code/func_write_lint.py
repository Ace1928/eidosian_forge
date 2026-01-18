from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
def write_lint(self) -> None:
    """Write lint results to stdout."""
    if self.summary:
        command = self.format_command()
        message = 'The test `%s` failed. See stderr output for details.' % command
        path = ''
        message = TestMessage(message, path)
        print(message)
    else:
        for message in self.messages:
            print(message)