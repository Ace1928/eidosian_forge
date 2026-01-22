import io
import logging
import os
import sys
import threading
import time
from io import StringIO
class TeeStream(object):

    def __init__(self, *ostreams, encoding=None):
        self.ostreams = ostreams
        self.encoding = encoding
        self._stdout = None
        self._stderr = None
        self._handles = []
        self._active_handles = []
        self._threads = []

    @property
    def STDOUT(self):
        if self._stdout is None:
            self._stdout = self.open(buffering=1)
        return self._stdout

    @property
    def STDERR(self):
        if self._stderr is None:
            self._stderr = self.open(buffering=0)
        return self._stderr

    def open(self, mode='w', buffering=-1, encoding=None, newline=None):
        if encoding is None:
            encoding = self.encoding
        handle = _StreamHandle(mode, buffering, encoding, newline)
        if handle.buffering:
            self._active_handles.append(handle)
        else:
            self._active_handles.insert(0, handle)
        self._handles.append(handle)
        self._start(handle)
        return handle.write_file

    def close(self, in_exception=False):
        for h in list(self._handles):
            h.close()
        _poll = _poll_interval
        while True:
            for th in self._threads:
                th.join(_poll)
            self._threads[:] = [th for th in self._threads if th.is_alive()]
            if not self._threads:
                break
            _poll *= 2
            if _poll_timeout <= _poll < 2 * _poll_timeout:
                if in_exception:
                    break
                logger.warning('Significant delay observed waiting to join reader threads, possible output stream deadlock')
            elif _poll >= _poll_timeout_deadlock:
                raise RuntimeError('TeeStream: deadlock observed joining reader threads')
        for h in list(self._handles):
            h.finalize(self.ostreams)
        self._threads.clear()
        self._handles.clear()
        self._active_handles.clear()
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        self.close(et is not None)

    def __del__(self):
        if threading.current_thread() not in self._threads:
            self.close()

    def _start(self, handle):
        if not _peek_available:
            th = threading.Thread(target=self._streamReader, args=(handle,))
            th.daemon = True
            th.start()
            self._threads.append(th)
        elif not self._threads:
            th = threading.Thread(target=self._mergedReader)
            th.daemon = True
            th.start()
            self._threads.append(th)
        else:
            pass

    def _streamReader(self, handle):
        while True:
            new_data = os.read(handle.read_pipe, io.DEFAULT_BUFFER_SIZE)
            if not new_data:
                break
            handle.decoder_buffer += new_data
            handle.decodeIncomingBuffer()
            handle.writeOutputBuffer(self.ostreams)

    def _mergedReader(self):
        noop = []
        handles = self._active_handles
        _poll = _poll_interval
        _fast_poll_ct = _poll_rampup
        new_data = ''
        while handles:
            if new_data is None:
                if _fast_poll_ct:
                    _fast_poll_ct -= 1
                    if not _fast_poll_ct:
                        _poll *= 10
                        if _poll < _poll_rampup_limit:
                            _fast_poll_ct = _poll_rampup
            else:
                new_data = None
            if _mswindows:
                for handle in list(handles):
                    try:
                        pipe = get_osfhandle(handle.read_pipe)
                        numAvail = PeekNamedPipe(pipe, 0)[1]
                        if numAvail:
                            result, new_data = ReadFile(pipe, numAvail, None)
                            handle.decoder_buffer += new_data
                            break
                    except:
                        handles.remove(handle)
                        new_data = None
                if new_data is None:
                    time.sleep(_poll)
                    continue
            else:
                ready_handles = select(list(handles), noop, noop, _poll)[0]
                if not ready_handles:
                    new_data = None
                    continue
                handle = ready_handles[0]
                new_data = os.read(handle.read_pipe, io.DEFAULT_BUFFER_SIZE)
                if not new_data:
                    handles.remove(handle)
                    continue
                handle.decoder_buffer += new_data
            handle.decodeIncomingBuffer()
            handle.writeOutputBuffer(self.ostreams)