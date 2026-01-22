from __future__ import annotations
import os
import os.path
import sys
from types import FrameType
from typing import Any, Iterable, Iterator
from coverage.exceptions import PluginError
from coverage.misc import isolate_module
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter
from coverage.types import (
class LabelledDebug:
    """A Debug writer, but with labels for prepending to the messages."""

    def __init__(self, label: str, debug: TDebugCtl, prev_labels: Iterable[str]=()):
        self.labels = list(prev_labels) + [label]
        self.debug = debug

    def add_label(self, label: str) -> LabelledDebug:
        """Add a label to the writer, and return a new `LabelledDebug`."""
        return LabelledDebug(label, self.debug, self.labels)

    def message_prefix(self) -> str:
        """The prefix to use on messages, combining the labels."""
        prefixes = self.labels + ['']
        return ':\n'.join(('  ' * i + label for i, label in enumerate(prefixes)))

    def write(self, message: str) -> None:
        """Write `message`, but with the labels prepended."""
        self.debug.write(f'{self.message_prefix()}{message}')