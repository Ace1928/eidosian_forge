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
class _Transfer(object):
    """Generic bits common to Uploads and Downloads."""

    def __init__(self, stream, close_stream=False, chunksize=None, auto_transfer=True, http=None, num_retries=5):
        self.__bytes_http = None
        self.__close_stream = close_stream
        self.__http = http
        self.__stream = stream
        self.__url = None
        self.__num_retries = 5
        self.num_retries = num_retries
        self.retry_func = http_wrapper.HandleExceptionsAndRebuildHttpConnections
        self.auto_transfer = auto_transfer
        self.chunksize = chunksize or 1048576

    def __repr__(self):
        return str(self)

    @property
    def close_stream(self):
        return self.__close_stream

    @property
    def http(self):
        return self.__http

    @property
    def bytes_http(self):
        return self.__bytes_http or self.http

    @bytes_http.setter
    def bytes_http(self, value):
        self.__bytes_http = value

    @property
    def num_retries(self):
        return self.__num_retries

    @num_retries.setter
    def num_retries(self, value):
        util.Typecheck(value, six.integer_types)
        if value < 0:
            raise exceptions.InvalidDataError('Cannot have negative value for num_retries')
        self.__num_retries = value

    @property
    def stream(self):
        return self.__stream

    @property
    def url(self):
        return self.__url

    def _Initialize(self, http, url):
        """Initialize this download by setting self.http and self.url.

        We want the user to be able to override self.http by having set
        the value in the constructor; in that case, we ignore the provided
        http.

        Args:
          http: An httplib2.Http instance or None.
          url: The url for this transfer.

        Returns:
          None. Initializes self.
        """
        self.EnsureUninitialized()
        if self.http is None:
            self.__http = http or http_wrapper.GetHttp()
        self.__url = url

    @property
    def initialized(self):
        return self.url is not None and self.http is not None

    @property
    def _type_name(self):
        return type(self).__name__

    def EnsureInitialized(self):
        if not self.initialized:
            raise exceptions.TransferInvalidError('Cannot use uninitialized %s' % self._type_name)

    def EnsureUninitialized(self):
        if self.initialized:
            raise exceptions.TransferInvalidError('Cannot re-initialize %s' % self._type_name)

    def __del__(self):
        if self.__close_stream:
            self.__stream.close()

    def _ExecuteCallback(self, callback, response):
        if callback is not None:
            threading.Thread(target=callback, args=(response, self)).start()