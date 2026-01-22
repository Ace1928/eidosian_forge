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
def __ProcessResponse(self, response):
    """Process response (by updating self and writing to self.stream)."""
    if response.status_code not in self._ACCEPTABLE_STATUSES:
        if response.status_code in (http_client.FORBIDDEN, http_client.NOT_FOUND):
            raise exceptions.HttpError.FromResponse(response)
        else:
            raise exceptions.TransferRetryError(response.content)
    if response.status_code in (http_client.OK, http_client.PARTIAL_CONTENT):
        try:
            self.stream.write(six.ensure_binary(response.content))
        except TypeError:
            self.stream.write(six.ensure_text(response.content))
        self.__progress += response.length
        if response.info and 'content-encoding' in response.info:
            self.__encoding = response.info['content-encoding']
    elif response.status_code == http_client.NO_CONTENT:
        self.stream.write('')
    return response