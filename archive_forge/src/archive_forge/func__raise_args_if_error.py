import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
def _raise_args_if_error(self, result_tuple):
    v1_error_codes = [b'norepository', b'NoSuchFile', b'FileExists', b'DirectoryNotEmpty', b'ShortReadvError', b'UnicodeEncodeError', b'UnicodeDecodeError', b'ReadOnlyError', b'nobranch', b'NoSuchRevision', b'nosuchrevision', b'LockContention', b'UnlockableTransport', b'LockFailed', b'TokenMismatch', b'ReadError', b'PermissionDenied']
    if result_tuple[0] in v1_error_codes:
        self._request.finished_reading()
        raise errors.ErrorFromSmartServer(result_tuple)