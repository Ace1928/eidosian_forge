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
@classmethod
def FromData(cls, stream, json_data, http, auto_transfer=None, gzip_encoded=False, client=None, **kwds):
    """Create a new Upload of stream from serialized json_data and http."""
    info = json.loads(json_data)
    missing_keys = cls._REQUIRED_SERIALIZATION_KEYS - set(info.keys())
    if missing_keys:
        raise exceptions.InvalidDataError('Invalid serialization data, missing keys: %s' % ', '.join(missing_keys))
    if 'total_size' in kwds:
        raise exceptions.InvalidUserInputError('Cannot override total_size on serialized Upload')
    upload = cls.FromStream(stream, info['mime_type'], total_size=info.get('total_size'), gzip_encoded=gzip_encoded, **kwds)
    if isinstance(stream, io.IOBase) and (not stream.seekable()):
        raise exceptions.InvalidUserInputError('Cannot restart resumable upload on non-seekable stream')
    if auto_transfer is not None:
        upload.auto_transfer = auto_transfer
    else:
        upload.auto_transfer = info['auto_transfer']
    if client is not None:
        url = client.FinalizeTransferUrl(info['url'])
    else:
        url = info['url']
    upload.strategy = RESUMABLE_UPLOAD
    upload._Initialize(http, url)
    upload.RefreshResumableUploadState()
    upload.EnsureInitialized()
    if upload.auto_transfer:
        upload.StreamInChunks()
    return upload