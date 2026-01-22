from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.core.updater import installers
def _update_absolute_position(self, offset):
    """Seeks to a position in the underlying stream.

    Catching up digesters sometimes requires seeking to a specific position in
    self._stream. Child classes wrap streams which are not seekable, and have
    different strategies to make it appear that a seek has occured, which can
    be supported by overriding this method.

    Args:
      offset (int): the position to seek to.

    Returns:
      the new position in the stream.
    """
    return self._stream.seek(offset)