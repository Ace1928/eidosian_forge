from __future__ import annotations
import heapq
import logging
import os
import sys
import time
import typing
import warnings
from contextlib import suppress
from urwid import display, signals
from urwid.command_map import Command, command_map
from urwid.display.common import INPUT_DESCRIPTORS_CHANGED
from urwid.util import StoppingContext, is_mouse_event
from urwid.widget import PopUpTarget
from .abstract_loop import ExitMainLoop
from .select_loop import SelectEventLoop
def watch_pipe(self, callback: Callable[[bytes], bool | None]) -> int:
    """
            Create a pipe for use by a subprocess or thread to trigger a callback
            in the process/thread running the main loop.

            :param callback: function taking one parameter to call from within the process/thread running the main loop
            :type callback: callable

            This method returns a file descriptor attached to the write end of a pipe.
            The read end of the pipe is added to the list of files :attr:`event_loop` is watching.
            When data is written to the pipe the callback function will be called
            and passed a single value containing data read from the pipe.

            This method may be used any time you want to update widgets from another thread or subprocess.

            Data may be written to the returned file descriptor with ``os.write(fd, data)``.
            Ensure that data is less than 512 bytes (or 4K on Linux)
            so that the callback will be triggered just once with the complete value of data passed in.

            If the callback returns ``False`` then the watch will be removed from :attr:`event_loop`
            and the read end of the pipe will be closed.
            You are responsible for closing the write end of the pipe with ``os.close(fd)``.
            """
    import fcntl
    pipe_rd, pipe_wr = os.pipe()
    fcntl.fcntl(pipe_rd, fcntl.F_SETFL, os.O_NONBLOCK)
    watch_handle = None

    def cb() -> None:
        data = os.read(pipe_rd, PIPE_BUFFER_READ_SIZE)
        if callback(data) is False:
            self.event_loop.remove_watch_file(watch_handle)
            os.close(pipe_rd)
    watch_handle = self.event_loop.watch_file(pipe_rd, cb)
    self._watch_pipes[pipe_wr] = (watch_handle, pipe_rd)
    return pipe_wr