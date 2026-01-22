import io
import logging
import os
import sys
import threading
import time
from io import StringIO
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