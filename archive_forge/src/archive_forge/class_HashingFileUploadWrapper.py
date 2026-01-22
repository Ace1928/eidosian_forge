from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import hashlib
import os
import six
from boto import config
import crcmod
from gslib.exception import CommandException
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import DEFAULT_FILE_BUFFER_SIZE
from gslib.utils.constants import MIN_SIZE_COMPUTE_LOGGING
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.constants import UTF8
class HashingFileUploadWrapper(object):
    """Wraps an input stream in a hash digester and exposes a stream interface.

  This class provides integrity checking during file uploads via the
  following properties:

  Calls to read will appropriately update digesters with all bytes read.
  Calls to seek (assuming it is supported by the wrapped stream) using
      os.SEEK_SET will catch up / reset the digesters to the specified
      position. If seek is called with a different os.SEEK mode, the caller
      must return to the original position using os.SEEK_SET before further
      reads.
  Calls to seek are fast if the desired position is equal to the position at
      the beginning of the last read call (we only need to re-hash bytes
      from that point on).
  """

    def __init__(self, stream, digesters, hash_algs, src_url, logger):
        """Initializes the wrapper.

    Args:
      stream: Input stream.
      digesters: dict of {string: hash digester} containing digesters, where
          string is the name of the hash algorithm.
      hash_algs: dict of {string: hash algorithm} for resetting and
          recalculating digesters. String is the name of the hash algorithm.
      src_url: Source FileUrl that is being copied.
      logger: For outputting log messages.
    """
        if not digesters:
            raise CommandException('HashingFileUploadWrapper used with no digesters.')
        elif not hash_algs:
            raise CommandException('HashingFileUploadWrapper used with no hash_algs.')
        self._orig_fp = stream
        self._digesters = digesters
        self._src_url = src_url
        self._logger = logger
        self._seek_away = None
        self._digesters_previous = {}
        for alg in self._digesters:
            self._digesters_previous[alg] = self._digesters[alg].copy()
        self._digesters_previous_mark = 0
        self._digesters_current_mark = 0
        self._hash_algs = hash_algs

    @property
    def mode(self):
        """Returns the mode of the underlying file descriptor, or None."""
        return getattr(self._orig_fp, 'mode', None)

    def read(self, size=-1):
        """"Reads from the wrapped file pointer and calculates hash digests.

    Args:
      size: The amount of bytes to read. If ommited or negative, the entire
          contents of the file will be read, hashed, and returned.

    Returns:
      Bytes from the wrapped stream.

    Raises:
      CommandException if the position of the wrapped stream is unknown.
    """
        if self._seek_away is not None:
            raise CommandException('Read called on hashing file pointer in an unknown position; cannot correctly compute digest.')
        data = self._orig_fp.read(size)
        if isinstance(data, six.text_type):
            data = data.encode(UTF8)
        self._digesters_previous_mark = self._digesters_current_mark
        for alg in self._digesters:
            self._digesters_previous[alg] = self._digesters[alg].copy()
            self._digesters[alg].update(data)
        self._digesters_current_mark += len(data)
        return data

    def tell(self):
        """Returns the current stream position."""
        return self._orig_fp.tell()

    def seekable(self):
        """Returns true if the stream is seekable."""
        return self._orig_fp.seekable()

    def seek(self, offset, whence=os.SEEK_SET):
        """Seeks in the wrapped file pointer and catches up hash digests.

    Args:
      offset: The offset to seek to.
      whence: os.SEEK_CUR, or SEEK_END, SEEK_SET.

    Returns:
      Return value from the wrapped stream's seek call.
    """
        if whence != os.SEEK_SET:
            self._seek_away = self._orig_fp.tell()
        else:
            self._seek_away = None
            if offset < self._digesters_previous_mark:
                for alg in self._digesters:
                    self._digesters[alg] = self._hash_algs[alg]()
                self._digesters_current_mark = 0
                self._orig_fp.seek(0)
                self._CatchUp(offset)
            elif offset == self._digesters_previous_mark:
                self._digesters_current_mark = self._digesters_previous_mark
                for alg in self._digesters:
                    self._digesters[alg] = self._digesters_previous[alg]
            elif offset < self._digesters_current_mark:
                self._digesters_current_mark = self._digesters_previous_mark
                for alg in self._digesters:
                    self._digesters[alg] = self._digesters_previous[alg]
                self._orig_fp.seek(self._digesters_previous_mark)
                self._CatchUp(offset - self._digesters_previous_mark)
            else:
                self._orig_fp.seek(self._digesters_current_mark)
                self._CatchUp(offset - self._digesters_current_mark)
        return self._orig_fp.seek(offset, whence)

    def _CatchUp(self, bytes_to_read):
        """Catches up hashes, but does not return data and uses little memory.

    Before calling this function, digesters_current_mark should be updated
    to the current location of the original stream and the self._digesters
    should be current to that point (but no further).

    Args:
      bytes_to_read: Number of bytes to catch up from the original stream.
    """
        if self._orig_fp.tell() != self._digesters_current_mark:
            raise CommandException('Invalid mark when catching up hashes. Stream position %s, hash position %s' % (self._orig_fp.tell(), self._digesters_current_mark))
        for alg in self._digesters:
            if bytes_to_read >= MIN_SIZE_COMPUTE_LOGGING:
                self._logger.debug('Catching up %s for %s...', alg, self._src_url.url_string)
            self._digesters_previous[alg] = self._digesters[alg].copy()
        self._digesters_previous_mark = self._digesters_current_mark
        bytes_remaining = bytes_to_read
        bytes_this_round = min(bytes_remaining, TRANSFER_BUFFER_SIZE)
        while bytes_this_round:
            data = self._orig_fp.read(bytes_this_round)
            if isinstance(data, six.text_type):
                data = data.encode(UTF8)
            bytes_remaining -= bytes_this_round
            for alg in self._digesters:
                self._digesters[alg].update(data)
            bytes_this_round = min(bytes_remaining, TRANSFER_BUFFER_SIZE)
        self._digesters_current_mark += bytes_to_read