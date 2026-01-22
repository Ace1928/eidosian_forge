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
class HttpMock(object):
    """Mock of httplib2.Http"""

    def __init__(self, filename=None, headers=None):
        """
    Args:
      filename: string, absolute filename to read response from
      headers: dict, header to return with response
    """
        if headers is None:
            headers = {'status': '200'}
        if filename:
            with open(filename, 'rb') as f:
                self.data = f.read()
        else:
            self.data = None
        self.response_headers = headers
        self.headers = None
        self.uri = None
        self.method = None
        self.body = None
        self.headers = None

    def request(self, uri, method='GET', body=None, headers=None, redirections=1, connection_type=None):
        self.uri = uri
        self.method = method
        self.body = body
        self.headers = headers
        return (httplib2.Response(self.response_headers), self.data)

    def close(self):
        return None