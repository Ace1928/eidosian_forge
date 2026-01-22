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