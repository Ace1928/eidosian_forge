from __future__ import annotations
import re
import contextlib
import io
import logging
import os
import signal
from subprocess import Popen, PIPE, DEVNULL
import subprocess
import threading
from textwrap import dedent
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import PathLike, Literal, TBD
class AutoInterrupt:
    """Process wrapper that terminates the wrapped process on finalization.

        This kills/interrupts the stored process instance once this instance goes out of
        scope. It is used to prevent processes piling up in case iterators stop reading.

        All attributes are wired through to the contained process object.

        The wait method is overridden to perform automatic status code checking and
        possibly raise.
        """
    __slots__ = ('proc', 'args', 'status')
    _status_code_if_terminate: int = 0

    def __init__(self, proc: Union[None, subprocess.Popen], args: Any) -> None:
        self.proc = proc
        self.args = args
        self.status: Union[int, None] = None

    def _terminate(self) -> None:
        """Terminate the underlying process."""
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        if proc.stdin:
            proc.stdin.close()
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
        try:
            if proc.poll() is not None:
                self.status = self._status_code_if_terminate or proc.poll()
                return
        except OSError as ex:
            _logger.info('Ignored error after process had died: %r', ex)
        if os is None or getattr(os, 'kill', None) is None:
            return
        try:
            proc.terminate()
            status = proc.wait()
            self.status = self._status_code_if_terminate or status
        except OSError as ex:
            _logger.info('Ignored error after process had died: %r', ex)

    def __del__(self) -> None:
        self._terminate()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.proc, attr)

    def wait(self, stderr: Union[None, str, bytes]=b'') -> int:
        """Wait for the process and return its status code.

            :param stderr: Previously read value of stderr, in case stderr is already closed.
            :warn: May deadlock if output or error pipes are used and not handled separately.
            :raise GitCommandError: If the return status is not 0.
            """
        if stderr is None:
            stderr_b = b''
        stderr_b = force_bytes(data=stderr, encoding='utf-8')
        status: Union[int, None]
        if self.proc is not None:
            status = self.proc.wait()
            p_stderr = self.proc.stderr
        else:
            status = self.status
            p_stderr = None

        def read_all_from_possibly_closed_stream(stream: Union[IO[bytes], None]) -> bytes:
            if stream:
                try:
                    return stderr_b + force_bytes(stream.read())
                except (OSError, ValueError):
                    return stderr_b or b''
            else:
                return stderr_b or b''
        if status != 0:
            errstr = read_all_from_possibly_closed_stream(p_stderr)
            _logger.debug('AutoInterrupt wait stderr: %r' % (errstr,))
            raise GitCommandError(remove_password_if_present(self.args), status, errstr)
        return status