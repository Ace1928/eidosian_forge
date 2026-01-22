from __future__ import print_function
import email.generator as email_generator
import email.mime.multipart as mime_multipart
import email.mime.nonmultipart as mime_nonmultipart
import io
import json
import mimetypes
import os
import threading
import six
from six.moves import http_client
from apitools.base.py import buffered_stream
from apitools.base.py import compression
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import stream_slice
from apitools.base.py import util
def __ComputeEndByte(self, start, end=None, use_chunks=True):
    """Compute the last byte to fetch for this request.

        This is all based on the HTTP spec for Range and
        Content-Range.

        Note that this is potentially confusing in several ways:
          * the value for the last byte is 0-based, eg "fetch 10 bytes
            from the beginning" would return 9 here.
          * if we have no information about size, and don't want to
            use the chunksize, we'll return None.
        See the tests for more examples.

        Args:
          start: byte to start at.
          end: (int or None, default: None) Suggested last byte.
          use_chunks: (bool, default: True) If False, ignore self.chunksize.

        Returns:
          Last byte to use in a Range header, or None.

        """
    end_byte = end
    if start < 0 and (not self.total_size):
        return end_byte
    if use_chunks:
        alternate = start + self.chunksize - 1
        if end_byte is not None:
            end_byte = min(end_byte, alternate)
        else:
            end_byte = alternate
    if self.total_size:
        alternate = self.total_size - 1
        if end_byte is not None:
            end_byte = min(end_byte, alternate)
        else:
            end_byte = alternate
    return end_byte