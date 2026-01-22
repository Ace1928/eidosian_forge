from hyperframe.exceptions import InvalidFrameError
from hyperframe.frame import (
from .exceptions import (
def _update_header_buffer(self, f):
    """
        Updates the internal header buffer. Returns a frame that should replace
        the current one. May throw exceptions if this frame is invalid.
        """
    if self._headers_buffer:
        stream_id = self._headers_buffer[0].stream_id
        valid_frame = f is not None and isinstance(f, ContinuationFrame) and (f.stream_id == stream_id)
        if not valid_frame:
            raise ProtocolError('Invalid frame during header block.')
        self._headers_buffer.append(f)
        if len(self._headers_buffer) > CONTINUATION_BACKLOG:
            raise ProtocolError('Too many continuation frames received.')
        if 'END_HEADERS' in f.flags:
            f = self._headers_buffer[0]
            f.flags.add('END_HEADERS')
            f.data = b''.join((x.data for x in self._headers_buffer))
            self._headers_buffer = []
        else:
            f = None
    elif isinstance(f, (HeadersFrame, PushPromiseFrame)) and 'END_HEADERS' not in f.flags:
        self._headers_buffer.append(f)
        f = None
    return f