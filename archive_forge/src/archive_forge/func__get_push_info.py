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
def _get_push_info(self, proc: 'Git.AutoInterrupt', progress: Union[Callable[..., Any], RemoteProgress, None], kill_after_timeout: Union[None, float]=None) -> PushInfoList:
    progress = to_progress_instance(progress)
    progress_handler = progress.new_message_handler()
    output: PushInfoList = PushInfoList()

    def stdout_handler(line: str) -> None:
        try:
            output.append(PushInfo._from_line(self, line))
        except ValueError:
            pass
    handle_process_output(proc, stdout_handler, progress_handler, finalizer=None, decode_streams=False, kill_after_timeout=kill_after_timeout)
    stderr_text = progress.error_lines and '\n'.join(progress.error_lines) or ''
    try:
        proc.wait(stderr=stderr_text)
    except Exception as e:
        if not output:
            raise
        elif stderr_text:
            _logger.warning('Error lines received while fetching: %s', stderr_text)
            output.error = e
    return output