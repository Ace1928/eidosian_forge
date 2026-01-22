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
Prints each of the url strings to stdout.

    Args:
      url_strings: String iterable.
      show_header: If true, print a header per file.
      start_byte: Starting byte of the file to print, used for constructing
                  range requests.
      end_byte: Ending byte of the file to print; used for constructing range
                requests. If this is negative, the start_byte is ignored and
                and end range is sent over HTTP (such as range: bytes -9)
      cat_out_fd: File descriptor to which output should be written. Defaults to
                 stdout if no file descriptor is supplied.
    Returns:
      0 on success.

    Raises:
      CommandException if no URLs can be found.
    