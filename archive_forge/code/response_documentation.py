import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
Read size bytes from the current position in the file.

        Reading across ranges is not supported. We rely on the underlying http
        client to clean the socket if we leave bytes unread. This may occur for
        the final boundary line of a multipart response or for any range
        request not entirely consumed by the client (due to offset coalescing)

        :param size:  The number of bytes to read.  Leave unspecified or pass
            -1 to read to EOF.
        