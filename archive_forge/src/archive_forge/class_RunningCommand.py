import asyncio
from collections import deque
import errno
import fcntl
import gc
import getpass
import glob as glob_module
import inspect
import logging
import os
import platform
import pty
import pwd
import re
import select
import signal
import stat
import struct
import sys
import termios
import textwrap
import threading
import time
import traceback
import tty
import warnings
import weakref
from asyncio import Queue as AQueue
from contextlib import contextmanager
from functools import partial
from importlib import metadata
from io import BytesIO, StringIO, UnsupportedOperation
from io import open as fdopen
from locale import getpreferredencoding
from queue import Empty, Queue
from shlex import quote as shlex_quote
from types import GeneratorType, ModuleType
from typing import Any, Dict, Type, Union
class RunningCommand(object):
    """this represents an executing Command object.  it is returned as the
    result of __call__() being executed on a Command instance.  this creates a
    reference to a OProc instance, which is a low-level wrapper around the
    process that was exec'd

    this is the class that gets manipulated the most by user code, and so it
    implements various convenience methods and logical mechanisms for the
    underlying process.  for example, if a user tries to access a
    backgrounded-process's stdout/err, the RunningCommand object is smart enough
    to know to wait() on the process to finish first.  and when the process
    finishes, RunningCommand is smart enough to translate exit codes to
    exceptions."""
    _OProc_attr_allowlist = {'signal', 'terminate', 'kill', 'kill_group', 'signal_group', 'pid', 'sid', 'pgid', 'ctty', 'input_thread_exc', 'output_thread_exc', 'bg_thread_exc'}

    def __init__(self, cmd, call_args, stdin, stdout, stderr):
        self.ran = ' '.join([shlex_quote(str(arg)) for arg in cmd])
        self.call_args = call_args
        self.cmd = cmd
        self.process = None
        self._waited_until_completion = False
        should_wait = True
        spawn_process = True
        self._force_noblock_iter = False
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.aio_output_complete = None
        else:
            self.aio_output_complete = asyncio.Event()
        self._stopped_iteration = False
        if call_args['with']:
            spawn_process = False
            get_prepend_stack().append(self)
        if call_args['piped'] or call_args['iter'] or call_args['iter_noblock']:
            should_wait = False
        if call_args['async']:
            should_wait = False
        if call_args['bg']:
            should_wait = False
        if call_args['err_to_out']:
            stderr = OProc.STDOUT
        done_callback = call_args['done']
        if done_callback:
            call_args['done'] = partial(done_callback, self)
        pipe = OProc.STDOUT
        if call_args['iter'] == 'out' or call_args['iter'] is True:
            pipe = OProc.STDOUT
        elif call_args['iter'] == 'err':
            pipe = OProc.STDERR
        if call_args['iter_noblock'] == 'out' or call_args['iter_noblock'] is True:
            pipe = OProc.STDOUT
        elif call_args['iter_noblock'] == 'err':
            pipe = OProc.STDERR
        self._spawned_and_waited = False
        if spawn_process:
            log_str_factory = call_args['log_msg'] or default_logger_str
            logger_str = log_str_factory(self.ran, call_args)
            self.log = Logger('command', logger_str)
            self.log.debug('starting process')
            if should_wait:
                self._spawned_and_waited = True
            process_assign_lock = threading.Lock()
            with process_assign_lock:
                self.process = OProc(self, self.log, cmd, stdin, stdout, stderr, self.call_args, pipe, process_assign_lock)
            logger_str = log_str_factory(self.ran, call_args, self.process.pid)
            self.log.context = self.log.sanitize_context(logger_str)
            self.log.info('process started')
            if should_wait:
                self.wait()

    def wait(self, timeout=None):
        """waits for the running command to finish.  this is called on all
        running commands, eventually, except for ones that run in the background

        if timeout is a number, it is the number of seconds to wait for the process to
        resolve. otherwise block on wait.

        this function can raise a TimeoutException, either because of a `_timeout` on
        the command itself as it was
        launched, or because of a timeout passed into this method.
        """
        if not self._waited_until_completion:
            if timeout is not None:
                waited_for = 0
                sleep_amt = 0.1
                alive = False
                exit_code = None
                if timeout < 0:
                    raise RuntimeError('timeout cannot be negative')
                while waited_for <= timeout:
                    alive, exit_code = self.process.is_alive()
                    if alive:
                        time.sleep(sleep_amt)
                        waited_for += sleep_amt
                    else:
                        break
                if alive:
                    raise TimeoutException(None, self.ran)
                self._waited_until_completion = True
            else:
                exit_code = self.process.wait()
                self._waited_until_completion = True
            if self.process.timed_out:
                raise TimeoutException(-exit_code, self.ran)
            else:
                self.handle_command_exit_code(exit_code)
                if self.process._stdin_process:
                    self.process._stdin_process.command.wait()
            self.log.debug('process completed')
        return self

    def is_alive(self):
        """returns whether or not we're still alive. this call has side-effects on
        OProc"""
        return self.process.is_alive()[0]

    def handle_command_exit_code(self, code):
        """here we determine if we had an exception, or an error code that we
        weren't expecting to see.  if we did, we create and raise an exception
        """
        ca = self.call_args
        exc_class = get_exc_exit_code_would_raise(code, ca['ok_code'], ca['piped'])
        if exc_class:
            exc = exc_class(self.ran, self.process.stdout, self.process.stderr, ca['truncate_exc'])
            raise exc

    @property
    def stdout(self):
        self.wait()
        return self.process.stdout

    @property
    def stderr(self):
        self.wait()
        return self.process.stderr

    @property
    def exit_code(self):
        self.wait()
        return self.process.exit_code

    def __len__(self):
        return len(str(self))

    def __enter__(self):
        """we don't actually do anything here because anything that should have
        been done would have been done in the Command.__call__ call.
        essentially all that has to happen is the command be pushed on the
        prepend stack."""
        pass

    def __iter__(self):
        return self

    def __next__(self):
        """allow us to iterate over the output of our command"""
        if self._stopped_iteration:
            raise StopIteration()
        pq = self.process._pipe_queue
        block_pq_read = not self._force_noblock_iter
        while True:
            try:
                chunk = pq.get(block_pq_read, self.call_args['iter_poll_time'])
            except Empty:
                if self.call_args['iter_noblock'] or self._force_noblock_iter:
                    return errno.EWOULDBLOCK
            else:
                if chunk is None:
                    self.wait()
                    self._stopped_iteration = True
                    raise StopIteration()
                try:
                    return chunk.decode(self.call_args['encoding'], self.call_args['decode_errors'])
                except UnicodeDecodeError:
                    return chunk

    def __await__(self):

        async def wait_for_completion():
            await self.aio_output_complete.wait()
            return str(self)
        return wait_for_completion().__await__()

    def __aiter__(self):
        self._aio_queue = AQueue(maxsize=1)
        self._force_noblock_iter = True

        async def queue_connector():
            try:
                for chunk in self:
                    if chunk == errno.EWOULDBLOCK:
                        await asyncio.sleep(0.01)
                    else:
                        await self._aio_queue.put(chunk)
            finally:
                await self._aio_queue.put(None)
        task = asyncio.create_task(queue_connector())
        self._aio_task = task
        return self

    async def __anext__(self):
        chunk = await self._aio_queue.get()
        if chunk is not None:
            return chunk
        else:
            exc = self._aio_task.exception()
            if exc is not None:
                raise exc
            raise StopAsyncIteration

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.call_args['with'] and get_prepend_stack():
            get_prepend_stack().pop()

    def __str__(self):
        if self.process and self.stdout:
            return self.stdout.decode(self.call_args['encoding'], self.call_args['decode_errors'])
        return ''

    def __eq__(self, other):
        return id(self) == id(other)

    def __contains__(self, item):
        return item in str(self)

    def __getattr__(self, p):
        if p in self._OProc_attr_allowlist:
            if self.process:
                return getattr(self.process, p)
            else:
                raise AttributeError
        if p in _unicode_methods:
            return getattr(str(self), p)
        raise AttributeError

    def __repr__(self):
        try:
            return str(self)
        except UnicodeDecodeError:
            if self.process:
                if self.stdout:
                    return repr(self.stdout)
            return repr('')

    def __long__(self):
        return int(str(self).strip())

    def __float__(self):
        return float(str(self).strip())

    def __int__(self):
        return int(str(self).strip())