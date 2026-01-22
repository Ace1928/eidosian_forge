import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def add_command(self, command):
    """Add a command to the command history.

    Args:
      command: The history command, as a str.

    Raises:
      TypeError: if command is not a str.
    """
    if self._commands and command == self._commands[-1]:
        return
    if not isinstance(command, str):
        raise TypeError('Attempt to enter non-str entry to command history')
    self._commands.append(command)
    if len(self._commands) > self._limit:
        self._commands = self._commands[-self._limit:]
    self._add_command_to_history_file(command)