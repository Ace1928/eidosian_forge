from __future__ import annotations
import re
import textwrap
import traceback
import typing as t
from .util import (
class FileDiff:
    """Parsed diff for a single file."""

    def __init__(self, old_path: str, new_path: str) -> None:
        self.old = DiffSide(old_path, new=False)
        self.new = DiffSide(new_path, new=True)
        self.headers: list[str] = []
        self.binary = False

    def append_header(self, line: str) -> None:
        """Append the given line to the list of headers for this file."""
        self.headers.append(line)

    @property
    def is_complete(self) -> bool:
        """True if the diff is complete, otherwise False."""
        return self.old.is_complete and self.new.is_complete