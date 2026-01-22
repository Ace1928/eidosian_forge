import socket
import re
import logging
import warnings
from requests.exceptions import RequestException, SSLError
import http.client as http_client
from urllib.parse import quote, unquote
from urllib.parse import urljoin, urlparse, urlunparse
from time import sleep, time
from swiftclient import version as swiftclient_version
from swiftclient.exceptions import ClientException
from swiftclient.requests_compat import SwiftClientRequestsSession
from swiftclient.utils import (
class _RetryBody(_ObjectBody):
    """
    Wrapper for object body response which triggers a retry
    (from offset) if the connection is dropped after partially
    downloading the object.
    """

    def __init__(self, resp, connection, container, obj, resp_chunk_size=None, query_string=None, response_dict=None, headers=None):
        """
        Wrap the underlying response

        :param resp: the response to wrap
        :param connection: Connection class instance
        :param container: the name of the container the object is in
        :param obj: the name of object we are downloading
        :param resp_chunk_size: if defined, chunk size of data to read
        :param query_string: if set will be appended with '?' to generated path
        :param response_dict: an optional dictionary into which to place
                         the response - status, reason and headers
        :param headers: an optional dictionary with additional headers to
                         include in the request
        """
        super(_RetryBody, self).__init__(resp, resp_chunk_size, None)
        self.expected_length = int(self.resp.getheader('Content-Length'))
        self.conn = connection
        self.container = container
        self.obj = obj
        self.query_string = query_string
        self.response_dict = response_dict
        self.headers = dict(headers) if headers is not None else {}
        self.bytes_read = 0

    def read(self, length=None):
        buf = None
        try:
            buf = self.resp.read(length)
            self.bytes_read += len(buf)
        except (socket.error, RequestException):
            if self.conn.attempts > self.conn.retries:
                raise
        if not buf and self.bytes_read < self.expected_length and (self.conn.attempts <= self.conn.retries):
            self.headers['Range'] = 'bytes=%d-' % self.bytes_read
            self.headers['If-Match'] = self.resp.getheader('ETag')
            hdrs, body = self.conn._retry(None, get_object, self.container, self.obj, resp_chunk_size=self.chunk_size, query_string=self.query_string, response_dict=self.response_dict, headers=self.headers, attempts=self.conn.attempts)
            expected_range = 'bytes %d-%d/%d' % (self.bytes_read, self.expected_length - 1, self.expected_length)
            if 'content-range' not in hdrs:
                logger.warning('Received 200 while retrying %s/%s; seeking...', self.container, self.obj)
                to_read = self.bytes_read
                while to_read > 0:
                    buf = body.resp.read(min(to_read, self.chunk_size))
                    to_read -= len(buf)
            elif hdrs['content-range'] != expected_range:
                msg = 'Expected range "%s" while retrying %s/%s but got "%s"' % (expected_range, self.container, self.obj, hdrs['content-range'])
                raise ClientException(msg)
            self.resp = body.resp
            buf = self.read(length)
        return buf