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
class MediaInMemoryUpload(MediaIoBaseUpload):
    """MediaUpload for a chunk of bytes.

  DEPRECATED: Use MediaIoBaseUpload with either io.TextIOBase or StringIO for
  the stream.
  """

    @util.positional(2)
    def __init__(self, body, mimetype='application/octet-stream', chunksize=DEFAULT_CHUNK_SIZE, resumable=False):
        """Create a new MediaInMemoryUpload.

  DEPRECATED: Use MediaIoBaseUpload with either io.TextIOBase or StringIO for
  the stream.

  Args:
    body: string, Bytes of body content.
    mimetype: string, Mime-type of the file or default of
      'application/octet-stream'.
    chunksize: int, File will be uploaded in chunks of this many bytes. Only
      used if resumable=True.
    resumable: bool, True if this is a resumable upload. False means upload
      in a single request.
    """
        fd = BytesIO(body)
        super(MediaInMemoryUpload, self).__init__(fd, mimetype, chunksize=chunksize, resumable=resumable)