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
class MultipartBytesGenerator(email_generator.BytesGenerator):
    """Generates a bytes Message object tree for multipart messages

        This is a BytesGenerator that has been modified to not attempt line
        termination character modification in the bytes payload. Known to
        work with the compat32 policy only. It may work on others, but not
        tested. The outfp object must accept bytes in its write method.
        """

    def _handle_text(self, msg):
        if msg._payload is None:
            return
        self.write(msg._payload)

    def _encode(self, s):
        return s.encode('ascii', 'surrogateescape')
    _writeBody = _handle_text