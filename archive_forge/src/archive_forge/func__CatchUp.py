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