import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
class AWSRequestPreparer:
    """
    This class performs preparation on AWSRequest objects similar to that of
    the PreparedRequest class does in the requests library. However, the logic
    has been boiled down to meet the specific use cases in botocore. Of note
    there are the following differences:
        This class does not heavily prepare the URL. Requests performed many
        validations and corrections to ensure the URL is properly formatted.
        Botocore either performs these validations elsewhere or otherwise
        consistently provides well formatted URLs.

        This class does not heavily prepare the body. Body preperation is
        simple and supports only the cases that we document: bytes and
        file-like objects to determine the content-length. This will also
        additionally prepare a body that is a dict to be url encoded params
        string as some signers rely on this. Finally, this class does not
        support multipart file uploads.

        This class does not prepare the method, auth or cookies.
    """

    def prepare(self, original):
        method = original.method
        url = self._prepare_url(original)
        body = self._prepare_body(original)
        headers = self._prepare_headers(original, body)
        stream_output = original.stream_output
        return AWSPreparedRequest(method, url, headers, body, stream_output)

    def _prepare_url(self, original):
        url = original.url
        if original.params:
            url_parts = urlparse(url)
            delim = '&' if url_parts.query else '?'
            if isinstance(original.params, Mapping):
                params_to_encode = list(original.params.items())
            else:
                params_to_encode = original.params
            params = urlencode(params_to_encode, doseq=True)
            url = delim.join((url, params))
        return url

    def _prepare_headers(self, original, prepared_body=None):
        headers = HeadersDict(original.headers.items())
        if 'Transfer-Encoding' in headers or 'Content-Length' in headers:
            return headers
        if original.method not in ('GET', 'HEAD', 'OPTIONS'):
            length = self._determine_content_length(prepared_body)
            if length is not None:
                headers['Content-Length'] = str(length)
            else:
                body_type = type(prepared_body)
                logger.debug('Failed to determine length of %s', body_type)
                headers['Transfer-Encoding'] = 'chunked'
        return headers

    def _to_utf8(self, item):
        key, value = item
        if isinstance(key, str):
            key = key.encode('utf-8')
        if isinstance(value, str):
            value = value.encode('utf-8')
        return (key, value)

    def _prepare_body(self, original):
        """Prepares the given HTTP body data."""
        body = original.data
        if body == b'':
            body = None
        if isinstance(body, dict):
            params = [self._to_utf8(item) for item in body.items()]
            body = urlencode(params, doseq=True)
        return body

    def _determine_content_length(self, body):
        return botocore.utils.determine_content_length(body)