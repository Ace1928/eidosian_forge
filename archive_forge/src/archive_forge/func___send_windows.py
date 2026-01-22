import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
def __send_windows(self, buf):
    if self._write_pending:
        try:
            nBytesWritten = winutils.get_overlapped_result(self.pipe, self._write, False)
            self._write_pending = False
        except pywintypes.error as e:
            if e.winerror == winutils.winerror.ERROR_IO_INCOMPLETE:
                self._read_pending = True
                return -errno.EAGAIN
            elif e.winerror in winutils.pipe_disconnected_errors:
                return -errno.ECONNRESET
            else:
                return -errno.EINVAL
    else:
        errCode, nBytesWritten = winutils.write_file(self.pipe, buf, self._write)
        if errCode:
            if errCode == winutils.winerror.ERROR_IO_PENDING:
                self._write_pending = True
                return -errno.EAGAIN
            if not nBytesWritten and errCode in winutils.pipe_disconnected_errors:
                return -errno.ECONNRESET
    return nBytesWritten