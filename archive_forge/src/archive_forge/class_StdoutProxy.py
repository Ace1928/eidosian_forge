from __future__ import annotations
import asyncio
import queue
import sys
import threading
import time
from contextlib import contextmanager
from typing import Generator, TextIO, cast
from .application import get_app_session, run_in_terminal
from .output import Output
class StdoutProxy:
    """
    File-like object, which prints everything written to it, output above the
    current application/prompt. This class is compatible with other file
    objects and can be used as a drop-in replacement for `sys.stdout` or can
    for instance be passed to `logging.StreamHandler`.

    The current application, above which we print, is determined by looking
    what application currently runs in the `AppSession` that is active during
    the creation of this instance.

    This class can be used as a context manager.

    In order to avoid having to repaint the prompt continuously for every
    little write, a short delay of `sleep_between_writes` seconds will be added
    between writes in order to bundle many smaller writes in a short timespan.
    """

    def __init__(self, sleep_between_writes: float=0.2, raw: bool=False) -> None:
        self.sleep_between_writes = sleep_between_writes
        self.raw = raw
        self._lock = threading.RLock()
        self._buffer: list[str] = []
        self.app_session = get_app_session()
        self._output: Output = self.app_session.output
        self._flush_queue: queue.Queue[str | _Done] = queue.Queue()
        self._flush_thread = self._start_write_thread()
        self.closed = False

    def __enter__(self) -> StdoutProxy:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """
        Stop `StdoutProxy` proxy.

        This will terminate the write thread, make sure everything is flushed
        and wait for the write thread to finish.
        """
        if not self.closed:
            self._flush_queue.put(_Done())
            self._flush_thread.join()
            self.closed = True

    def _start_write_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self._write_thread, name='patch-stdout-flush-thread', daemon=True)
        thread.start()
        return thread

    def _write_thread(self) -> None:
        done = False
        while not done:
            item = self._flush_queue.get()
            if isinstance(item, _Done):
                break
            if not item:
                continue
            text = []
            text.append(item)
            while True:
                try:
                    item = self._flush_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    if isinstance(item, _Done):
                        done = True
                    else:
                        text.append(item)
            app_loop = self._get_app_loop()
            self._write_and_flush(app_loop, ''.join(text))
            if app_loop is not None:
                time.sleep(self.sleep_between_writes)

    def _get_app_loop(self) -> asyncio.AbstractEventLoop | None:
        """
        Return the event loop for the application currently running in our
        `AppSession`.
        """
        app = self.app_session.app
        if app is None:
            return None
        return app.loop

    def _write_and_flush(self, loop: asyncio.AbstractEventLoop | None, text: str) -> None:
        """
        Write the given text to stdout and flush.
        If an application is running, use `run_in_terminal`.
        """

        def write_and_flush() -> None:
            self._output.enable_autowrap()
            if self.raw:
                self._output.write_raw(text)
            else:
                self._output.write(text)
            self._output.flush()

        def write_and_flush_in_loop() -> None:
            run_in_terminal(write_and_flush, in_executor=False)
        if loop is None:
            write_and_flush()
        else:
            loop.call_soon_threadsafe(write_and_flush_in_loop)

    def _write(self, data: str) -> None:
        """
        Note: print()-statements cause to multiple write calls.
              (write('line') and write('
')). Of course we don't want to call
              `run_in_terminal` for every individual call, because that's too
              expensive, and as long as the newline hasn't been written, the
              text itself is again overwritten by the rendering of the input
              command line. Therefor, we have a little buffer which holds the
              text until a newline is written to stdout.
        """
        if '\n' in data:
            before, after = data.rsplit('\n', 1)
            to_write = self._buffer + [before, '\n']
            self._buffer = [after]
            text = ''.join(to_write)
            self._flush_queue.put(text)
        else:
            self._buffer.append(data)

    def _flush(self) -> None:
        text = ''.join(self._buffer)
        self._buffer = []
        self._flush_queue.put(text)

    def write(self, data: str) -> int:
        with self._lock:
            self._write(data)
        return len(data)

    def flush(self) -> None:
        """
        Flush buffered output.
        """
        with self._lock:
            self._flush()

    @property
    def original_stdout(self) -> TextIO:
        return self._output.stdout or sys.__stdout__

    def fileno(self) -> int:
        return self._output.fileno()

    def isatty(self) -> bool:
        stdout = self._output.stdout
        if stdout is None:
            return False
        return stdout.isatty()

    @property
    def encoding(self) -> str:
        return self._output.encoding()

    @property
    def errors(self) -> str:
        return 'strict'