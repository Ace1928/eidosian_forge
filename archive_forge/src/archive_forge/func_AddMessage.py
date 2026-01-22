from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
def AddMessage(self, message, indentation_level=0):
    """Adds a MultilineConsoleMessage to the MultilineConsoleOutput object.

    Args:
      message: str, The message that will be displayed.
      indentation_level: int, The indentation level of the message. Each
        indentation is represented by two spaces.

    Returns:
      MultilineConsoleMessage, a message object that can be used to dynamically
      change the printed message.
    """
    with self._lock:
        return self._AddMessage(message, indentation_level=indentation_level)