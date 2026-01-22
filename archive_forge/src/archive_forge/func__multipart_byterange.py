import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def _multipart_byterange(self, data, offset, boundary, file_size=b'*'):
    """Encode a part of a file as a multipart/byterange MIME type.

        When a range request is issued, the HTTP response body can be
        decomposed in parts, each one representing a range (start, size) in a
        file.

        :param data: The payload.
        :param offset: where data starts in the file
        :param boundary: used to separate the parts
        :param file_size: the size of the file containing the range (default to
            '*' meaning unknown)

        :return: a string containing the data encoded as it will appear in the
            HTTP response body.
        """
    bline = self._boundary_line()
    range = bline
    if isinstance(file_size, int):
        file_size = b'%d' % file_size
    range += b'Content-Range: bytes %d-%d/%s\r\n' % (offset, offset + len(data) - 1, file_size)
    range += b'\r\n'
    range += data
    return range