from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.command_lib.storage import thread_messages
Sends operation progress information to global status queue.

    Args:
      current_byte (int): Index of byte being operated on.
      *args (list[any]): Unused.
    