import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
class FileFromURLWrapper(object):
    """File from URL wrapper.

    The :class:`FileFromURLWrapper` object gives you the ability to stream file
    from provided URL in chunks by :class:`MultipartEncoder`.
    Provide a stateless solution for streaming file from one server to another.
    You can use the :class:`FileFromURLWrapper` without a session or with
    a session as demonstated by the examples below:

    .. code-block:: python
        # no session

        import requests
        from requests_toolbelt import MultipartEncoder, FileFromURLWrapper

        url = 'https://httpbin.org/image/png'
        streaming_encoder = MultipartEncoder(
            fields={
                'file': FileFromURLWrapper(url)
            }
        )
        r = requests.post(
            'https://httpbin.org/post', data=streaming_encoder,
            headers={'Content-Type': streaming_encoder.content_type}
        )

    .. code-block:: python
        # using a session

        import requests
        from requests_toolbelt import MultipartEncoder, FileFromURLWrapper

        session = requests.Session()
        url = 'https://httpbin.org/image/png'
        streaming_encoder = MultipartEncoder(
            fields={
                'file': FileFromURLWrapper(url, session=session)
            }
        )
        r = session.post(
            'https://httpbin.org/post', data=streaming_encoder,
            headers={'Content-Type': streaming_encoder.content_type}
        )

    """

    def __init__(self, file_url, session=None):
        self.session = session or requests.Session()
        requested_file = self._request_for_file(file_url)
        self.len = int(requested_file.headers['content-length'])
        self.raw_data = requested_file.raw

    def _request_for_file(self, file_url):
        """Make call for file under provided URL."""
        response = self.session.get(file_url, stream=True)
        content_length = response.headers.get('content-length', None)
        if content_length is None:
            error_msg = 'Data from provided URL {url} is not supported. Lack of content-length Header in requested file response.'.format(url=file_url)
            raise FileNotSupportedError(error_msg)
        elif not content_length.isdigit():
            error_msg = 'Data from provided URL {url} is not supported. content-length header value is not a digit.'.format(url=file_url)
            raise FileNotSupportedError(error_msg)
        return response

    def read(self, chunk_size):
        """Read file in chunks."""
        chunk_size = chunk_size if chunk_size >= 0 else self.len
        chunk = self.raw_data.read(chunk_size) or b''
        self.len -= len(chunk) if chunk else 0
        return chunk