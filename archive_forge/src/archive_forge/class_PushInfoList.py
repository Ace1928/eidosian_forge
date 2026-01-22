import contextlib
import logging
import re
from git.cmd import Git, handle_process_output
from git.compat import defenc, force_text
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import GitCommandError
from git.refs import Head, Reference, RemoteReference, SymbolicReference, TagReference
from git.util import (
from typing import (
from git.types import PathLike, Literal, Commit_ish
class PushInfoList(IterableList[PushInfo]):
    """IterableList of PushInfo objects."""

    def __new__(cls) -> 'PushInfoList':
        return cast(PushInfoList, IterableList.__new__(cls, 'push_infos'))

    def __init__(self) -> None:
        super().__init__('push_infos')
        self.error: Optional[Exception] = None

    def raise_if_error(self) -> None:
        """Raise an exception if any ref failed to push."""
        if self.error:
            raise self.error