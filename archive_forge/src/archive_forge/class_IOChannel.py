import warnings
import sys
import socket
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..module import get_introspection_module
from .._gi import (variant_type_from_string, source_new,
from ..overrides import override, deprecated, deprecated_attr
from gi import PyGIDeprecationWarning, version_info
from gi import _option as option
from gi import _gi
from gi._error import GError
class IOChannel(GLib.IOChannel):

    def __new__(cls, filedes=None, filename=None, mode=None, hwnd=None):
        if filedes is not None:
            return GLib.IOChannel.unix_new(filedes)
        if filename is not None:
            return GLib.IOChannel.new_file(filename, mode or 'r')
        if hwnd is not None:
            return GLib.IOChannel.win32_new_fd(hwnd)
        raise TypeError('either a valid file descriptor, file name, or window handle must be supplied')

    def __init__(self, *args, **kwargs):
        return super(IOChannel, self).__init__()

    def read(self, max_count=-1):
        return io_channel_read(self, max_count)

    def readline(self, size_hint=-1):
        status, buf, length, terminator_pos = self.read_line()
        if buf is None:
            return ''
        return buf

    def readlines(self, size_hint=-1):
        lines = []
        status = GLib.IOStatus.NORMAL
        while status == GLib.IOStatus.NORMAL:
            status, buf, length, terminator_pos = self.read_line()
            if buf is None:
                buf = ''
            lines.append(buf)
        return lines

    def write(self, buf, buflen=-1):
        if not isinstance(buf, bytes):
            buf = buf.encode('UTF-8')
        if buflen == -1:
            buflen = len(buf)
        status, written = self.write_chars(buf, buflen)
        return written

    def writelines(self, lines):
        for line in lines:
            self.write(line)
    _whence_map = {0: GLib.SeekType.SET, 1: GLib.SeekType.CUR, 2: GLib.SeekType.END}

    def seek(self, offset, whence=0):
        try:
            w = self._whence_map[whence]
        except KeyError:
            raise ValueError("invalid 'whence' value")
        return self.seek_position(offset, w)

    def add_watch(self, condition, callback, *user_data, **kwargs):
        priority = kwargs.get('priority', GLib.PRIORITY_DEFAULT)
        return io_add_watch(self, priority, condition, callback, *user_data)
    add_watch = deprecated(add_watch, 'GLib.io_add_watch()')

    def __iter__(self):
        return self

    def __next__(self):
        status, buf, length, terminator_pos = self.read_line()
        if status == GLib.IOStatus.NORMAL:
            return buf
        raise StopIteration
    next = __next__