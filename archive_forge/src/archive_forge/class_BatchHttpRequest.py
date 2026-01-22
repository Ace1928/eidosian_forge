from __future__ import absolute_import
import six
from six.moves import http_client
from six.moves import range
from six import BytesIO, StringIO
from six.moves.urllib.parse import urlparse, urlunparse, quote, unquote
import copy
import httplib2
import json
import logging
import mimetypes
import os
import random
import socket
import time
import uuid
from email.generator import Generator
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
from email.parser import FeedParser
from googleapiclient import _helpers as util
from googleapiclient import _auth
from googleapiclient.errors import BatchError
from googleapiclient.errors import HttpError
from googleapiclient.errors import InvalidChunkSizeError
from googleapiclient.errors import ResumableUploadError
from googleapiclient.errors import UnexpectedBodyError
from googleapiclient.errors import UnexpectedMethodError
from googleapiclient.model import JsonModel
class BatchHttpRequest(object):
    '''Batches multiple HttpRequest objects into a single HTTP request.

  Example:
    from googleapiclient.http import BatchHttpRequest

    def list_animals(request_id, response, exception):
      """Do something with the animals list response."""
      if exception is not None:
        # Do something with the exception.
        pass
      else:
        # Do something with the response.
        pass

    def list_farmers(request_id, response, exception):
      """Do something with the farmers list response."""
      if exception is not None:
        # Do something with the exception.
        pass
      else:
        # Do something with the response.
        pass

    service = build('farm', 'v2')

    batch = BatchHttpRequest()

    batch.add(service.animals().list(), list_animals)
    batch.add(service.farmers().list(), list_farmers)
    batch.execute(http=http)
  '''

    @util.positional(1)
    def __init__(self, callback=None, batch_uri=None):
        """Constructor for a BatchHttpRequest.

    Args:
      callback: callable, A callback to be called for each response, of the
        form callback(id, response, exception). The first parameter is the
        request id, and the second is the deserialized response object. The
        third is an googleapiclient.errors.HttpError exception object if an HTTP error
        occurred while processing the request, or None if no error occurred.
      batch_uri: string, URI to send batch requests to.
    """
        if batch_uri is None:
            batch_uri = _LEGACY_BATCH_URI
        if batch_uri == _LEGACY_BATCH_URI:
            LOGGER.warning('You have constructed a BatchHttpRequest using the legacy batch endpoint %s. This endpoint will be turned down on August 12, 2020. Please provide the API-specific endpoint or use service.new_batch_http_request(). For more details see https://developers.googleblog.com/2018/03/discontinuing-support-for-json-rpc-and.htmland https://developers.google.com/api-client-library/python/guide/batch.', _LEGACY_BATCH_URI)
        self._batch_uri = batch_uri
        self._callback = callback
        self._requests = {}
        self._callbacks = {}
        self._order = []
        self._last_auto_id = 0
        self._base_id = None
        self._responses = {}
        self._refreshed_credentials = {}

    def _refresh_and_apply_credentials(self, request, http):
        """Refresh the credentials and apply to the request.

    Args:
      request: HttpRequest, the request.
      http: httplib2.Http, the global http object for the batch.
    """
        creds = None
        request_credentials = False
        if request.http is not None:
            creds = _auth.get_credentials_from_http(request.http)
            request_credentials = True
        if creds is None and http is not None:
            creds = _auth.get_credentials_from_http(http)
        if creds is not None:
            if id(creds) not in self._refreshed_credentials:
                _auth.refresh_credentials(creds)
                self._refreshed_credentials[id(creds)] = 1
        if request.http is None or not request_credentials:
            _auth.apply_credentials(creds, request.headers)

    def _id_to_header(self, id_):
        """Convert an id to a Content-ID header value.

    Args:
      id_: string, identifier of individual request.

    Returns:
      A Content-ID header with the id_ encoded into it. A UUID is prepended to
      the value because Content-ID headers are supposed to be universally
      unique.
    """
        if self._base_id is None:
            self._base_id = uuid.uuid4()
        return '<%s + %s>' % (self._base_id, quote(id_))

    def _header_to_id(self, header):
        """Convert a Content-ID header value to an id.

    Presumes the Content-ID header conforms to the format that _id_to_header()
    returns.

    Args:
      header: string, Content-ID header value.

    Returns:
      The extracted id value.

    Raises:
      BatchError if the header is not in the expected format.
    """
        if header[0] != '<' or header[-1] != '>':
            raise BatchError('Invalid value for Content-ID: %s' % header)
        if '+' not in header:
            raise BatchError('Invalid value for Content-ID: %s' % header)
        base, id_ = header[1:-1].split(' + ', 1)
        return unquote(id_)

    def _serialize_request(self, request):
        """Convert an HttpRequest object into a string.

    Args:
      request: HttpRequest, the request to serialize.

    Returns:
      The request as a string in application/http format.
    """
        parsed = urlparse(request.uri)
        request_line = urlunparse(('', '', parsed.path, parsed.params, parsed.query, ''))
        status_line = request.method + ' ' + request_line + ' HTTP/1.1\n'
        major, minor = request.headers.get('content-type', 'application/json').split('/')
        msg = MIMENonMultipart(major, minor)
        headers = request.headers.copy()
        if request.http is not None:
            credentials = _auth.get_credentials_from_http(request.http)
            if credentials is not None:
                _auth.apply_credentials(credentials, headers)
        if 'content-type' in headers:
            del headers['content-type']
        for key, value in six.iteritems(headers):
            msg[key] = value
        msg['Host'] = parsed.netloc
        msg.set_unixfrom(None)
        if request.body is not None:
            msg.set_payload(request.body)
            msg['content-length'] = str(len(request.body))
        fp = StringIO()
        g = Generator(fp, maxheaderlen=0)
        g.flatten(msg, unixfrom=False)
        body = fp.getvalue()
        return status_line + body

    def _deserialize_response(self, payload):
        """Convert string into httplib2 response and content.

    Args:
      payload: string, headers and body as a string.

    Returns:
      A pair (resp, content), such as would be returned from httplib2.request.
    """
        status_line, payload = payload.split('\n', 1)
        protocol, status, reason = status_line.split(' ', 2)
        parser = FeedParser()
        parser.feed(payload)
        msg = parser.close()
        msg['status'] = status
        resp = httplib2.Response(msg)
        resp.reason = reason
        resp.version = int(protocol.split('/', 1)[1].replace('.', ''))
        content = payload.split('\r\n\r\n', 1)[1]
        return (resp, content)

    def _new_id(self):
        """Create a new id.

    Auto incrementing number that avoids conflicts with ids already used.

    Returns:
       string, a new unique id.
    """
        self._last_auto_id += 1
        while str(self._last_auto_id) in self._requests:
            self._last_auto_id += 1
        return str(self._last_auto_id)

    @util.positional(2)
    def add(self, request, callback=None, request_id=None):
        """Add a new request.

    Every callback added will be paired with a unique id, the request_id. That
    unique id will be passed back to the callback when the response comes back
    from the server. The default behavior is to have the library generate it's
    own unique id. If the caller passes in a request_id then they must ensure
    uniqueness for each request_id, and if they are not an exception is
    raised. Callers should either supply all request_ids or never supply a
    request id, to avoid such an error.

    Args:
      request: HttpRequest, Request to add to the batch.
      callback: callable, A callback to be called for this response, of the
        form callback(id, response, exception). The first parameter is the
        request id, and the second is the deserialized response object. The
        third is an googleapiclient.errors.HttpError exception object if an HTTP error
        occurred while processing the request, or None if no errors occurred.
      request_id: string, A unique id for the request. The id will be passed
        to the callback with the response.

    Returns:
      None

    Raises:
      BatchError if a media request is added to a batch.
      KeyError is the request_id is not unique.
    """
        if len(self._order) >= MAX_BATCH_LIMIT:
            raise BatchError('Exceeded the maximum calls(%d) in a single batch request.' % MAX_BATCH_LIMIT)
        if request_id is None:
            request_id = self._new_id()
        if request.resumable is not None:
            raise BatchError('Media requests cannot be used in a batch request.')
        if request_id in self._requests:
            raise KeyError('A request with this ID already exists: %s' % request_id)
        self._requests[request_id] = request
        self._callbacks[request_id] = callback
        self._order.append(request_id)

    def _execute(self, http, order, requests):
        """Serialize batch request, send to server, process response.

    Args:
      http: httplib2.Http, an http object to be used to make the request with.
      order: list, list of request ids in the order they were added to the
        batch.
      requests: list, list of request objects to send.

    Raises:
      httplib2.HttpLib2Error if a transport error has occurred.
      googleapiclient.errors.BatchError if the response is the wrong format.
    """
        message = MIMEMultipart('mixed')
        setattr(message, '_write_headers', lambda self: None)
        for request_id in order:
            request = requests[request_id]
            msg = MIMENonMultipart('application', 'http')
            msg['Content-Transfer-Encoding'] = 'binary'
            msg['Content-ID'] = self._id_to_header(request_id)
            body = self._serialize_request(request)
            msg.set_payload(body)
            message.attach(msg)
        fp = StringIO()
        g = Generator(fp, mangle_from_=False)
        g.flatten(message, unixfrom=False)
        body = fp.getvalue()
        headers = {}
        headers['content-type'] = 'multipart/mixed; boundary="%s"' % message.get_boundary()
        resp, content = http.request(self._batch_uri, method='POST', body=body, headers=headers)
        if resp.status >= 300:
            raise HttpError(resp, content, uri=self._batch_uri)
        header = 'content-type: %s\r\n\r\n' % resp['content-type']
        if six.PY3:
            content = content.decode('utf-8')
        for_parser = header + content
        parser = FeedParser()
        parser.feed(for_parser)
        mime_response = parser.close()
        if not mime_response.is_multipart():
            raise BatchError('Response not in multipart/mixed format.', resp=resp, content=content)
        for part in mime_response.get_payload():
            request_id = self._header_to_id(part['Content-ID'])
            response, content = self._deserialize_response(part.get_payload())
            if isinstance(content, six.text_type):
                content = content.encode('utf-8')
            self._responses[request_id] = (response, content)

    @util.positional(1)
    def execute(self, http=None):
        """Execute all the requests as a single batched HTTP request.

    Args:
      http: httplib2.Http, an http object to be used in place of the one the
        HttpRequest request object was constructed with. If one isn't supplied
        then use a http object from the requests in this batch.

    Returns:
      None

    Raises:
      httplib2.HttpLib2Error if a transport error has occurred.
      googleapiclient.errors.BatchError if the response is the wrong format.
    """
        if len(self._order) == 0:
            return None
        if http is None:
            for request_id in self._order:
                request = self._requests[request_id]
                if request is not None:
                    http = request.http
                    break
        if http is None:
            raise ValueError('Missing a valid http object.')
        creds = _auth.get_credentials_from_http(http)
        if creds is not None:
            if not _auth.is_valid(creds):
                LOGGER.info('Attempting refresh to obtain initial access_token')
                _auth.refresh_credentials(creds)
        self._execute(http, self._order, self._requests)
        redo_requests = {}
        redo_order = []
        for request_id in self._order:
            resp, content = self._responses[request_id]
            if resp['status'] == '401':
                redo_order.append(request_id)
                request = self._requests[request_id]
                self._refresh_and_apply_credentials(request, http)
                redo_requests[request_id] = request
        if redo_requests:
            self._execute(http, redo_order, redo_requests)
        for request_id in self._order:
            resp, content = self._responses[request_id]
            request = self._requests[request_id]
            callback = self._callbacks[request_id]
            response = None
            exception = None
            try:
                if resp.status >= 300:
                    raise HttpError(resp, content, uri=request.uri)
                response = request.postproc(resp, content)
            except HttpError as e:
                exception = e
            if callback is not None:
                callback(request_id, response, exception)
            if self._callback is not None:
                self._callback(request_id, response, exception)