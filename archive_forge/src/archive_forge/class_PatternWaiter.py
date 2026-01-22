import asyncio
import errno
import signal
from sys import version_info as py_version_info
from pexpect import EOF
class PatternWaiter(asyncio.Protocol):
    transport = None

    def set_expecter(self, expecter):
        self.expecter = expecter
        self.fut = asyncio.Future()

    def found(self, result):
        if not self.fut.done():
            self.fut.set_result(result)
            self.transport.pause_reading()

    def error(self, exc):
        if not self.fut.done():
            self.fut.set_exception(exc)
            self.transport.pause_reading()

    def connection_made(self, transport):
        self.transport = transport

    def data_received(self, data):
        spawn = self.expecter.spawn
        s = spawn._decoder.decode(data)
        spawn._log(s, 'read')
        if self.fut.done():
            spawn._before.write(s)
            spawn._buffer.write(s)
            return
        try:
            index = self.expecter.new_data(s)
            if index is not None:
                self.found(index)
        except Exception as exc:
            self.expecter.errored()
            self.error(exc)

    def eof_received(self):
        try:
            self.expecter.spawn.flag_eof = True
            index = self.expecter.eof()
        except EOF as exc:
            self.error(exc)
        else:
            self.found(index)

    def connection_lost(self, exc):
        if isinstance(exc, OSError) and exc.errno == errno.EIO:
            self.eof_received()
        elif exc is not None:
            self.error(exc)