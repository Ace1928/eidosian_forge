from hyperframe.exceptions import InvalidFrameError
from hyperframe.frame import (
from .exceptions import (
def _parse_frame_header(self, data):
    """
        Parses the frame header from the data. Either returns a tuple of
        (frame, length), or throws an exception. The returned frame may be None
        if the frame is of unknown type.
        """
    try:
        frame, length = Frame.parse_frame_header(data[:9])
    except ValueError as e:
        raise ProtocolError('Invalid frame header received: %s' % str(e))
    return (frame, length)