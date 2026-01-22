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