from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
def write_console(self) -> None:
    """Write results to console."""
    if self.summary:
        display.error(self.summary)
    else:
        if self.python_version:
            specifier = ' on python %s' % self.python_version
        else:
            specifier = ''
        display.error('Found %d %s issue(s)%s which need to be resolved:' % (len(self.messages), self.test or self.command, specifier))
        for message in self.messages:
            display.error(message.format(show_confidence=True))
        doc_url = self.find_docs()
        if doc_url:
            display.info('See documentation for help: %s' % doc_url)