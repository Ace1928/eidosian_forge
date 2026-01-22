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
class MediaIoBaseUpload(MediaUpload):
    """A MediaUpload for a io.Base objects.

  Note that the Python file object is compatible with io.Base and can be used
  with this class also.

    fh = BytesIO('...Some data to upload...')
    media = MediaIoBaseUpload(fh, mimetype='image/png',
      chunksize=1024*1024, resumable=True)
    farm.animals().insert(
        id='cow',
        name='cow.png',
        media_body=media).execute()

  Depending on the platform you are working on, you may pass -1 as the
  chunksize, which indicates that the entire file should be uploaded in a single
  request. If the underlying platform supports streams, such as Python 2.6 or
  later, then this can be very efficient as it avoids multiple connections, and
  also avoids loading the entire file into memory before sending it. Note that
  Google App Engine has a 5MB limit on request size, so you should never set
  your chunksize larger than 5MB, or to -1.
  """

    @util.positional(3)
    def __init__(self, fd, mimetype, chunksize=DEFAULT_CHUNK_SIZE, resumable=False):
        """Constructor.

    Args:
      fd: io.Base or file object, The source of the bytes to upload. MUST be
        opened in blocking mode, do not use streams opened in non-blocking mode.
        The given stream must be seekable, that is, it must be able to call
        seek() on fd.
      mimetype: string, Mime-type of the file.
      chunksize: int, File will be uploaded in chunks of this many bytes. Only
        used if resumable=True. Pass in a value of -1 if the file is to be
        uploaded as a single chunk. Note that Google App Engine has a 5MB limit
        on request size, so you should never set your chunksize larger than 5MB,
        or to -1.
      resumable: bool, True if this is a resumable upload. False means upload
        in a single request.
    """
        super(MediaIoBaseUpload, self).__init__()
        self._fd = fd
        self._mimetype = mimetype
        if not (chunksize == -1 or chunksize > 0):
            raise InvalidChunkSizeError()
        self._chunksize = chunksize
        self._resumable = resumable
        self._fd.seek(0, os.SEEK_END)
        self._size = self._fd.tell()

    def chunksize(self):
        """Chunk size for resumable uploads.

    Returns:
      Chunk size in bytes.
    """
        return self._chunksize

    def mimetype(self):
        """Mime type of the body.

    Returns:
      Mime type.
    """
        return self._mimetype

    def size(self):
        """Size of upload.

    Returns:
      Size of the body, or None of the size is unknown.
    """
        return self._size

    def resumable(self):
        """Whether this upload is resumable.

    Returns:
      True if resumable upload or False.
    """
        return self._resumable

    def getbytes(self, begin, length):
        """Get bytes from the media.

    Args:
      begin: int, offset from beginning of file.
      length: int, number of bytes to read, starting at begin.

    Returns:
      A string of bytes read. May be shorted than length if EOF was reached
      first.
    """
        self._fd.seek(begin)
        return self._fd.read(length)

    def has_stream(self):
        """Does the underlying upload support a streaming interface.

    Streaming means it is an io.IOBase subclass that supports seek, i.e.
    seekable() returns True.

    Returns:
      True if the call to stream() will return an instance of a seekable io.Base
      subclass.
    """
        return True

    def stream(self):
        """A stream interface to the data being uploaded.

    Returns:
      The returned value is an io.IOBase subclass that supports seek, i.e.
      seekable() returns True.
    """
        return self._fd

    def to_json(self):
        """This upload type is not serializable."""
        raise NotImplementedError('MediaIoBaseUpload is not serializable.')