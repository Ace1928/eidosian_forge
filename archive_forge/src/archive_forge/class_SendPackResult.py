import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
class SendPackResult:
    """Result of a upload-pack operation.

    Attributes:
      refs: Dictionary with all remote refs
      agent: User agent string
      ref_status: Optional dictionary mapping ref name to error message (if it
        failed to update), or None if it was updated successfully
    """
    _FORWARDED_ATTRS: ClassVar[Set[str]] = {'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values', 'viewitems', 'viewkeys', 'viewvalues'}

    def __init__(self, refs, agent=None, ref_status=None) -> None:
        self.refs = refs
        self.agent = agent
        self.ref_status = ref_status

    def _warn_deprecated(self):
        import warnings
        warnings.warn('Use SendPackResult.refs instead.', DeprecationWarning, stacklevel=3)

    def __eq__(self, other):
        if isinstance(other, dict):
            self._warn_deprecated()
            return self.refs == other
        return self.refs == other.refs and self.agent == other.agent

    def __contains__(self, name) -> bool:
        self._warn_deprecated()
        return name in self.refs

    def __getitem__(self, name):
        self._warn_deprecated()
        return self.refs[name]

    def __len__(self) -> int:
        self._warn_deprecated()
        return len(self.refs)

    def __iter__(self):
        self._warn_deprecated()
        return iter(self.refs)

    def __getattribute__(self, name):
        if name in type(self)._FORWARDED_ATTRS:
            self._warn_deprecated()
            return getattr(self.refs, name)
        return super().__getattribute__(name)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.refs!r}, {self.agent!r})'