from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
class ResponseHandler:
    """Abstract base class for an object that handles a smart response."""

    def read_response_tuple(self, expect_body=False):
        """Reads and returns the response tuple for the current request.

        :keyword expect_body: a boolean indicating if a body is expected in the
            response.  Some protocol versions needs this information to know
            when a response is finished.  If False, read_body_bytes should
            *not* be called afterwards.  Defaults to False.
        :returns: tuple of response arguments.
        """
        raise NotImplementedError(self.read_response_tuple)

    def read_body_bytes(self, count=-1):
        """Read and return some bytes from the body.

        :param count: if specified, read up to this many bytes.  By default,
            reads the entire body.
        :returns: str of bytes from the response body.
        """
        raise NotImplementedError(self.read_body_bytes)

    def read_streamed_body(self):
        """Returns an iterable that reads and returns a series of body chunks.
        """
        raise NotImplementedError(self.read_streamed_body)

    def cancel_read_body(self):
        """Stop expecting a body for this response.

        If expect_body was passed to read_response_tuple, this cancels that
        expectation (and thus finishes reading the response, allowing a new
        request to be issued).  This is useful if a response turns out to be an
        error rather than a normal result with a body.
        """
        raise NotImplementedError(self.cancel_read_body)