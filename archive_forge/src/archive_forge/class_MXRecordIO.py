from collections import namedtuple
from multiprocessing import current_process
import ctypes
import struct
import numbers
import numpy as np
from .base import _LIB
from .base import RecordIOHandle
from .base import check_call
from .base import c_str
class MXRecordIO(object):
    """Reads/writes `RecordIO` data format, supporting sequential read and write.

    Examples
    ---------
    >>> record = mx.recordio.MXRecordIO('tmp.rec', 'w')
    <mxnet.recordio.MXRecordIO object at 0x10ef40ed0>
    >>> for i in range(5):
    ...    record.write('record_%d'%i)
    >>> record.close()
    >>> record = mx.recordio.MXRecordIO('tmp.rec', 'r')
    >>> for i in range(5):
    ...    item = record.read()
    ...    print(item)
    record_0
    record_1
    record_2
    record_3
    record_4
    >>> record.close()

    Parameters
    ----------
    uri : string
        Path to the record file.
    flag : string
        'w' for write or 'r' for read.
    """

    def __init__(self, uri, flag):
        self.uri = c_str(uri)
        self.handle = RecordIOHandle()
        self.flag = flag
        self.pid = None
        self.is_open = False
        self.open()

    def open(self):
        """Opens the record file."""
        if self.flag == 'w':
            check_call(_LIB.MXRecordIOWriterCreate(self.uri, ctypes.byref(self.handle)))
            self.writable = True
        elif self.flag == 'r':
            check_call(_LIB.MXRecordIOReaderCreate(self.uri, ctypes.byref(self.handle)))
            self.writable = False
        else:
            raise ValueError('Invalid flag %s' % self.flag)
        self.pid = current_process().pid
        self.is_open = True

    def __del__(self):
        self.close()

    def __getstate__(self):
        """Override pickling behavior."""
        is_open = self.is_open
        self.close()
        d = dict(self.__dict__)
        d['is_open'] = is_open
        uri = self.uri.value
        try:
            uri = uri.decode('utf-8')
        except AttributeError:
            pass
        del d['handle']
        d['uri'] = uri
        return d

    def __setstate__(self, d):
        """Restore from pickled."""
        self.__dict__ = d
        is_open = d['is_open']
        self.is_open = False
        self.handle = RecordIOHandle()
        self.uri = c_str(self.uri)
        if is_open:
            self.open()

    def _check_pid(self, allow_reset=False):
        """Check process id to ensure integrity, reset if in new process."""
        if not self.pid == current_process().pid:
            if allow_reset:
                self.reset()
            else:
                raise RuntimeError('Forbidden operation in multiple processes')

    def close(self):
        """Closes the record file."""
        if not self.is_open:
            return
        if self.writable:
            check_call(_LIB.MXRecordIOWriterFree(self.handle))
        else:
            check_call(_LIB.MXRecordIOReaderFree(self.handle))
        self.is_open = False
        self.pid = None

    def reset(self):
        """Resets the pointer to first item.

        If the record is opened with 'w', this function will truncate the file to empty.

        Examples
        ---------
        >>> record = mx.recordio.MXRecordIO('tmp.rec', 'r')
        >>> for i in range(2):
        ...    item = record.read()
        ...    print(item)
        record_0
        record_1
        >>> record.reset()  # Pointer is reset.
        >>> print(record.read()) # Started reading from start again.
        record_0
        >>> record.close()
        """
        self.close()
        self.open()

    def write(self, buf):
        """Inserts a string buffer as a record.

        Examples
        ---------
        >>> record = mx.recordio.MXRecordIO('tmp.rec', 'w')
        >>> for i in range(5):
        ...    record.write('record_%d'%i)
        >>> record.close()

        Parameters
        ----------
        buf : string (python2), bytes (python3)
            Buffer to write.
        """
        assert self.writable
        self._check_pid(allow_reset=False)
        check_call(_LIB.MXRecordIOWriterWriteRecord(self.handle, ctypes.c_char_p(buf), ctypes.c_size_t(len(buf))))

    def read(self):
        """Returns record as a string.

        Examples
        ---------
        >>> record = mx.recordio.MXRecordIO('tmp.rec', 'r')
        >>> for i in range(5):
        ...    item = record.read()
        ...    print(item)
        record_0
        record_1
        record_2
        record_3
        record_4
        >>> record.close()

        Returns
        ----------
        buf : string
            Buffer read.
        """
        assert not self.writable
        self._check_pid(allow_reset=False)
        buf = ctypes.c_char_p()
        size = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOReaderReadRecord(self.handle, ctypes.byref(buf), ctypes.byref(size)))
        if buf:
            buf = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char * size.value))
            return buf.contents.raw
        else:
            return None