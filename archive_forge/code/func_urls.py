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
@property
def urls(self) -> Iterator[str]:
    """:return: Iterator yielding all configured URL targets on a remote as strings"""
    try:
        remote_details = self.repo.git.remote('get-url', '--all', self.name)
        assert isinstance(remote_details, str)
        for line in remote_details.split('\n'):
            yield line
    except GitCommandError as ex:
        if 'Unknown subcommand: get-url' in str(ex):
            try:
                remote_details = self.repo.git.remote('show', self.name)
                assert isinstance(remote_details, str)
                for line in remote_details.split('\n'):
                    if '  Push  URL:' in line:
                        yield line.split(': ')[-1]
            except GitCommandError as _ex:
                if any((msg in str(_ex) for msg in ['correct access rights', 'cannot run ssh'])):
                    remote_details = self.repo.git.config('--get-all', 'remote.%s.url' % self.name)
                    assert isinstance(remote_details, str)
                    for line in remote_details.split('\n'):
                        yield line
                else:
                    raise _ex
        else:
            raise ex