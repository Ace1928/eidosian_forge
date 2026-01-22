import io
import logging
import os.path
import urllib.parse
from smart_open import bytebuffer, constants
import smart_open.utils
class SeekableBufferedInputBase(BufferedInputBase):
    """
    Implement seekable streamed reader from a web site.
    Supports Kerberos, client certificate and Basic HTTP authentication.
    """

    def __init__(self, url, mode='r', buffer_size=DEFAULT_BUFFER_SIZE, kerberos=False, user=None, password=None, cert=None, headers=None, timeout=None):
        """
        If Kerberos is True, will attempt to use the local Kerberos credentials.
        If cert is set, will try to use a client certificate
        Otherwise, will try to use "basic" HTTP authentication via username/password.

        If none of those are set, will connect unauthenticated.
        """
        self.url = url
        if kerberos:
            import requests_kerberos
            self.auth = requests_kerberos.HTTPKerberosAuth()
        elif user is not None and password is not None:
            self.auth = (user, password)
        else:
            self.auth = None
        if headers is None:
            self.headers = _HEADERS.copy()
        else:
            self.headers = headers
        self.cert = cert
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.mode = mode
        self.response = self._partial_request()
        if not self.response.ok:
            self.response.raise_for_status()
        logger.debug('self.response: %r, raw: %r', self.response, self.response.raw)
        self.content_length = int(self.response.headers.get('Content-Length', -1))
        self._seekable = self.response.headers.get('Accept-Ranges', '').lower() != 'none'
        self._read_iter = self.response.iter_content(self.buffer_size)
        self._read_buffer = bytebuffer.ByteBuffer(buffer_size)
        self._current_pos = 0
        self.raw = None

    def seek(self, offset, whence=0):
        """Seek to the specified position.

        :param int offset: The offset in bytes.
        :param int whence: Where the offset is from.

        Returns the position after seeking."""
        logger.debug('seeking to offset: %r whence: %r', offset, whence)
        if whence not in constants.WHENCE_CHOICES:
            raise ValueError('invalid whence, expected one of %r' % constants.WHENCE_CHOICES)
        if not self.seekable():
            raise OSError('stream is not seekable')
        if whence == constants.WHENCE_START:
            new_pos = offset
        elif whence == constants.WHENCE_CURRENT:
            new_pos = self._current_pos + offset
        elif whence == constants.WHENCE_END:
            new_pos = self.content_length + offset
        if self.content_length == -1:
            new_pos = smart_open.utils.clamp(new_pos, maxval=None)
        else:
            new_pos = smart_open.utils.clamp(new_pos, maxval=self.content_length)
        if self._current_pos == new_pos:
            return self._current_pos
        logger.debug('http seeking from current_pos: %d to new_pos: %d', self._current_pos, new_pos)
        self._current_pos = new_pos
        if new_pos == self.content_length:
            self.response = None
            self._read_iter = None
            self._read_buffer.empty()
        else:
            response = self._partial_request(new_pos)
            if response.ok:
                self.response = response
                self._read_iter = self.response.iter_content(self.buffer_size)
                self._read_buffer.empty()
            else:
                self.response = None
        return self._current_pos

    def tell(self):
        return self._current_pos

    def seekable(self, *args, **kwargs):
        return self._seekable

    def truncate(self, size=None):
        """Unsupported."""
        raise io.UnsupportedOperation

    def _partial_request(self, start_pos=None):
        if start_pos is not None:
            self.headers.update({'range': smart_open.utils.make_range_string(start_pos)})
        response = requests.get(self.url, auth=self.auth, stream=True, cert=self.cert, headers=self.headers, timeout=self.timeout)
        return response