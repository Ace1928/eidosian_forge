from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import io
import sys
from boto import config
from gslib.cloud_api import EncryptionException
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.storage_url import StorageUrlFromString
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.encryption_helper import FindMatchingCSEKInBotoConfig
from gslib.utils.metadata_util import ObjectIsGzipEncoded
from gslib.utils import text_util
def _WriteBytesBufferedFileToFile(self, src_fd, dst_fd):
    """Copies contents of the source to the destination via buffered IO.

    Buffered reads are necessary in the case where you're reading from a
    source that produces more data than can fit into memory all at once. This
    method does not close either file when finished.

    Args:
      src_fd: The already-open source file to read from.
      dst_fd: The already-open destination file to write to.
    """
    while True:
        buf = src_fd.read(io.DEFAULT_BUFFER_SIZE)
        if not buf:
            break
        text_util.write_to_fd(dst_fd, buf)