import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def _open_body(self, start=None, stop=None):
    """Open a connection to download the specified range of bytes. Store
        the open file handle in self._body.

        If no range is specified, start defaults to self._position.
        start and stop follow the semantics of the http range header,
        so a stop without a start will read bytes beginning at stop.

        As a side effect, set self._content_length. Set self._position
        to self._content_length if start is past end of file.
        """
    if start is None and stop is None:
        start = self._position
    range_string = smart_open.utils.make_range_string(start, stop)
    try:
        response = _get(self._client, self._bucket, self._key, self._version_id, range_string)
    except IOError as ioe:
        error_response = _unwrap_ioerror(ioe)
        if error_response is None or error_response.get('Code') != _OUT_OF_RANGE:
            raise
        try:
            self._position = self._content_length = int(error_response['ActualObjectSize'])
            self._body = io.BytesIO()
        except KeyError:
            response = _get(self._client, self._bucket, self._key, self._version_id, None)
            self._position = self._content_length = response['ContentLength']
            self._body = response['Body']
    else:
        logger.debug('%s: RetryAttempts: %d', self, response['ResponseMetadata']['RetryAttempts'])
        _, start, stop, length = smart_open.utils.parse_content_range(response['ContentRange'])
        self._content_length = length
        self._position = start
        self._body = response['Body']