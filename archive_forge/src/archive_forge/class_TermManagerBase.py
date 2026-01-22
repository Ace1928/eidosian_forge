from __future__ import annotations
import asyncio
import codecs
import itertools
import logging
import os
import select
import signal
import warnings
from collections import deque
from concurrent import futures
from typing import TYPE_CHECKING, Any, Coroutine
from tornado.ioloop import IOLoop
class TermManagerBase:
    """Base class for a terminal manager."""

    def __init__(self, shell_command: str, server_url: str='', term_settings: Any=None, extra_env: Any=None, ioloop: Any=None, blocking_io_executor: Any=None):
        """Initialize the manager."""
        self.shell_command = shell_command
        self.server_url = server_url
        self.term_settings = term_settings or {}
        self.extra_env = extra_env
        self.log = logging.getLogger(__name__)
        self.ptys_by_fd: dict[int, PtyWithClients] = {}
        if blocking_io_executor is None:
            self._blocking_io_executor_is_external = False
            self.blocking_io_executor = futures.ThreadPoolExecutor(max_workers=1)
        else:
            self._blocking_io_executor_is_external = True
            self.blocking_io_executor = blocking_io_executor
        if ioloop is not None:
            warnings.warn(f'Setting {self.__class__.__name__}.ioloop is deprecated and ignored', DeprecationWarning, stacklevel=2)

    def make_term_env(self, height: int=25, width: int=80, winheight: int=0, winwidth: int=0, **kwargs: Any) -> dict[str, str]:
        """Build the environment variables for the process in the terminal."""
        env = os.environ.copy()
        env['TERM'] = self.term_settings.get('type', DEFAULT_TERM_TYPE)
        dimensions = '%dx%d' % (width, height)
        if winwidth and winheight:
            dimensions += ';%dx%d' % (winwidth, winheight)
        env[ENV_PREFIX + 'DIMENSIONS'] = dimensions
        env['COLUMNS'] = str(width)
        env['LINES'] = str(height)
        if self.server_url:
            env[ENV_PREFIX + 'URL'] = self.server_url
        if self.extra_env:
            _update_removing(env, self.extra_env)
        term_env = kwargs.get('extra_env', {})
        if term_env and isinstance(term_env, dict):
            _update_removing(env, term_env)
        return env

    def new_terminal(self, **kwargs: Any) -> PtyWithClients:
        """Make a new terminal, return a :class:`PtyWithClients` instance."""
        options = self.term_settings.copy()
        options['shell_command'] = self.shell_command
        options.update(kwargs)
        argv = options['shell_command']
        env = self.make_term_env(**options)
        cwd = options.get('cwd', None)
        return PtyWithClients(argv, env, cwd)

    def start_reading(self, ptywclients: PtyWithClients) -> None:
        """Connect a terminal to the tornado event loop to read data from it."""
        fd = ptywclients.ptyproc.fd
        self.ptys_by_fd[fd] = ptywclients
        loop = IOLoop.current()
        loop.add_handler(fd, self.pty_read, loop.READ)

    def on_eof(self, ptywclients: PtyWithClients) -> None:
        """Called when the pty has closed."""
        fd = ptywclients.ptyproc.fd
        self.log.info('EOF on FD %d; stopping reading', fd)
        del self.ptys_by_fd[fd]
        IOLoop.current().remove_handler(fd)
        ptywclients.ptyproc.close()

    def pty_read(self, fd: int, events: Any=None) -> None:
        """Called by the event loop when there is pty data ready to read."""
        if not _poll(fd, timeout=0.1):
            self.log.debug('Spurious pty_read() on fd %s', fd)
            return
        ptywclients = self.ptys_by_fd[fd]
        try:
            self.pre_pty_read_hook(ptywclients)
            s = ptywclients.ptyproc.read(65536)
            ptywclients.read_buffer.append(s)
            for client in ptywclients.clients:
                client.on_pty_read(s)
        except EOFError:
            self.on_eof(ptywclients)
            for client in ptywclients.clients:
                client.on_pty_died()

    def pre_pty_read_hook(self, ptywclients: PtyWithClients) -> None:
        """Hook before pty read, subclass can patch something into ptywclients when pty_read"""

    def get_terminal(self, url_component: Any=None) -> PtyWithClients:
        """Override in a subclass to give a terminal to a new websocket connection

        The :class:`TermSocket` handler works with zero or one URL components
        (capturing groups in the URL spec regex). If it receives one, it is
        passed as the ``url_component`` parameter; otherwise, this is None.
        """
        raise NotImplementedError

    def client_disconnected(self, websocket: Any) -> None:
        """Override this to e.g. kill terminals on client disconnection."""

    async def shutdown(self) -> None:
        """Shutdown the manager."""
        await self.kill_all()
        if not self._blocking_io_executor_is_external:
            self.blocking_io_executor.shutdown(wait=False, cancel_futures=True)

    async def kill_all(self) -> None:
        """Kill all terminals."""
        futures = []
        for term in self.ptys_by_fd.values():
            futures.append(term.terminate(force=True))
        if futures:
            await asyncio.gather(*futures)