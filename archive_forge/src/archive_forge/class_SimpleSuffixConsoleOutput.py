from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
class SimpleSuffixConsoleOutput(ConsoleOutput):
    """A simple, suffix-only implementation of ConsoleOutput.

  In this context, simple means that only updating the last line is supported.
  This means that this is supported in all ASCII environments as it only relies
  on carriage returns ('\\r') for modifying output. Suffix-only means that only
  modifying the ending of messages is supported, either via a
  detail_message_callback or by modifying the suffix of a SuffixConsoleMessage.
  """

    def __init__(self, stream):
        """Constructor.

    Args:
      stream: The output stream to write to.
    """
        self._stream = stream
        self._messages = []
        self._last_print_index = 0
        self._lock = threading.Lock()
        super(SimpleSuffixConsoleOutput, self).__init__()

    def AddMessage(self, message, detail_message_callback=None, indentation_level=0):
        """Adds a SuffixConsoleMessage to the SimpleSuffixConsoleOutput object.

    Args:
      message: str, The message that will be displayed.
      detail_message_callback: func() -> str, A no argument function that will
        be called and the result will be appended to the message on each call
        to UpdateConsole.
      indentation_level: int, The indentation level of the message. Each
        indentation is represented by two spaces.

    Returns:
      SuffixConsoleMessage, a message object that can be used to dynamically
      change the printed message.
    """
        with self._lock:
            return self._AddMessage(message, detail_message_callback=detail_message_callback, indentation_level=indentation_level)

    def _AddMessage(self, message, detail_message_callback=None, indentation_level=0):
        console_message = SuffixConsoleMessage(message, self._stream, detail_message_callback=detail_message_callback, indentation_level=indentation_level)
        self._messages.append(console_message)
        return console_message

    def UpdateMessage(self, message, new_suffix):
        """Updates the suffix of the given SuffixConsoleMessage."""
        if not message:
            raise ValueError('A message must be passed.')
        if message not in self._messages:
            raise ValueError('The given message does not belong to this output object.')
        if self._messages and message != self._messages[-1]:
            raise ValueError('Only the last added message can be updated.')
        with self._lock:
            message._UpdateSuffix(new_suffix)

    def UpdateConsole(self):
        with self._lock:
            self._UpdateConsole()

    def _UpdateConsole(self):
        """Updates the console output to show any updated or added messages."""
        if self._messages:
            if self._last_print_index < len(self._messages) - 1:
                for message in self._messages[self._last_print_index:-1]:
                    message.Print()
                    self._stream.write('\n')
                self._last_print_index = len(self._messages) - 1
            self._messages[self._last_print_index].Print()